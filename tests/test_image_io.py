"""Unit tests for the low-level multi-channel I/O and geometry helpers.

These primitives (``tiff_size``, ``_match_border_value``, ``_pad_constant``)
are exercised transitively by the end-to-end training tests, but those tests
take many minutes per parametrization. Covering the helpers directly here keeps
regressions in the multi-channel plumbing cheap to diagnose.
"""

import tarfile
from io import BytesIO

import numpy as np
import pytest
import tifffile

from PIL import Image

from ro_yolov7.utils import image_ops
from ro_yolov7.utils.image_io import (
    IMREAD_COLOR,
    IMREAD_GRAYSCALE,
    IMREAD_UNCHANGED,
    imdecode,
    imread,
    imwrite,
    tiff_size,
)
from ro_yolov7.utils.datasets import (
    LoadImagesAndLabels,
    _match_border_value,
    _pad_constant,
    build_yolo_tar_shard_samples,
    is_tar_shard_yolo_subset_dir,
    read_tar_member_bytes,
    split_tar_member_ref,
)


# ---------------------------------------------------------------------------
# tiff_size
# ---------------------------------------------------------------------------


def _write_multisample_tiff(path, img):
    """Write a multi-sample TIFF the way the rest of the project does."""
    n = img.shape[-1] if img.ndim == 3 else 1
    if img.ndim == 3 and n not in (1, 3, 4):
        if n == 2:
            kwargs = dict(photometric="minisblack", extrasamples=["unassalpha"])
        else:
            kwargs = dict(
                photometric="rgb",
                extrasamples=["unassalpha"] * (n - 3),
            )
        tifffile.imwrite(str(path), img, **kwargs)
    else:
        tifffile.imwrite(str(path), img)


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5])
def test_tiff_size_returns_logical_width_height(tmp_path, channels):
    h, w = 48, 80  # deliberately non-square so swaps are visible
    if channels == 1:
        img = np.zeros((h, w), dtype=np.uint8)
    else:
        img = np.zeros((h, w, channels), dtype=np.uint8)

    path = tmp_path / f"img_{channels}c.tif"
    _write_multisample_tiff(path, img)

    assert tiff_size(str(path)) == (w, h)


def test_tiff_size_accepts_bytes_stream(tmp_path):
    img = np.zeros((30, 50, 5), dtype=np.uint8)
    path = tmp_path / "img_5c.tif"
    _write_multisample_tiff(path, img)

    with open(path, "rb") as f:
        data = f.read()

    assert tiff_size(BytesIO(data)) == (50, 30)


# ---------------------------------------------------------------------------
# imread / imwrite round-trip for multi-channel TIFFs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5])
def test_imwrite_imread_roundtrip_multichannel_tiff(tmp_path, channels):
    h, w = 24, 32
    if channels == 1:
        img = np.arange(h * w, dtype=np.uint8).reshape(h, w)
    else:
        img = np.zeros((h, w, channels), dtype=np.uint8)
        for c in range(channels):
            img[..., c] = (c + 1) * 17  # distinct per channel

    path = tmp_path / f"roundtrip_{channels}c.tif"

    if channels in (1, 3, 4) or channels == 1:
        assert imwrite(str(path), img)
    else:
        # imwrite uses tifffile under the hood, which can't infer photometric
        # for 2/5 channel arrays. Use the explicit helper, same as the dataset
        # tests do.
        _write_multisample_tiff(path, img)

    loaded = imread(str(path))
    assert loaded is not None
    if channels == 1:
        assert loaded.shape == (h, w)
    else:
        assert loaded.shape == (h, w, channels)
    np.testing.assert_array_equal(loaded, img)


# ---------------------------------------------------------------------------
# _match_border_value
# ---------------------------------------------------------------------------


def _img_with_channels(channels):
    if channels == 1:
        return np.zeros((4, 4), dtype=np.uint8)
    return np.zeros((4, 4, channels), dtype=np.uint8)


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5, 7])
def test_match_border_value_scalar_fills_all_channels(channels):
    bv = _match_border_value(114, _img_with_channels(channels))
    assert bv == tuple([114] * channels)


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5])
def test_match_border_value_matching_tuple_passes_through(channels):
    color = tuple(range(10, 10 + channels))
    bv = _match_border_value(color, _img_with_channels(channels))
    assert bv == color


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5, 6])
def test_match_border_value_single_element_tuple_expands(channels):
    bv = _match_border_value((42,), _img_with_channels(channels))
    assert bv == tuple([42] * channels)


def test_match_border_value_3tuple_on_5_channel_image_pads_with_first():
    bv = _match_border_value((114, 114, 114), _img_with_channels(5))
    assert bv == (114, 114, 114, 114, 114)


def test_match_border_value_5tuple_on_3_channel_image_truncates():
    bv = _match_border_value((1, 2, 3, 4, 5), _img_with_channels(3))
    assert bv == (1, 2, 3)


def test_match_border_value_float_scalar_is_intified():
    bv = _match_border_value(114.0, _img_with_channels(3))
    assert bv == (114, 114, 114)


# ---------------------------------------------------------------------------
# _pad_constant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5, 6, 7, 8, 9])
def test_pad_constant_matches_per_channel_pad(channels):
    rng = np.random.default_rng(seed=channels)
    img = rng.integers(0, 256, size=(6, 8, channels), dtype=np.uint8)
    bv = tuple(range(100, 100 + channels))

    out = _pad_constant(img, 2, 3, 1, 4, bv)

    expected = np.stack(
        [np.pad(img[..., c], ((2, 3), (1, 4)), constant_values=bv[c]) for c in range(channels)],
        axis=-1,
    )
    np.testing.assert_array_equal(out, expected)


def test_pad_constant_2d_image():
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = _pad_constant(img, 1, 1, 1, 1, (7,))
    np.testing.assert_array_equal(out, np.pad(img, 1, constant_values=7))


# ---------------------------------------------------------------------------
# imread / imdecode flags and channel order for non-TIFF files
# ---------------------------------------------------------------------------


def _bgr_test_image():
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    img[..., 0] = 200  # blue
    img[..., 1] = 100  # green
    img[..., 2] = 50  # red
    img[2:5, 3:7] = (10, 20, 30)
    return img


def test_imwrite_png_is_rgb_on_disk_and_bgr_in_memory(tmp_path):
    img = _bgr_test_image()
    path = tmp_path / "bgr.png"
    assert imwrite(str(path), img)

    np.testing.assert_array_equal(np.asarray(Image.open(path)), img[..., ::-1])
    np.testing.assert_array_equal(imread(str(path)), img)
    np.testing.assert_array_equal(imread(str(path), IMREAD_UNCHANGED), img)


def test_imread_grayscale_of_colour_png_uses_luma_weights(tmp_path):
    img = _bgr_test_image()
    path = tmp_path / "bgr.png"
    imwrite(str(path), img)

    gray = imread(str(path), IMREAD_GRAYSCALE)
    b, g, r = (img[..., i].astype(np.float64) for i in range(3))
    expected = np.rint(0.299 * r + 0.587 * g + 0.114 * b)
    assert gray.shape == img.shape[:2]
    np.testing.assert_allclose(gray, expected, atol=1)


def test_imread_colour_of_grayscale_png_repeats_the_channel(tmp_path):
    img = np.arange(120, dtype=np.uint8).reshape(10, 12)
    path = tmp_path / "gray.png"
    assert imwrite(str(path), img)

    colour = imread(str(path), IMREAD_COLOR)
    assert colour.shape == (10, 12, 3)
    for c in range(3):
        np.testing.assert_array_equal(colour[..., c], img)
    np.testing.assert_array_equal(imread(str(path), IMREAD_UNCHANGED), img)


def test_imread_16_bit_png(tmp_path):
    img = (np.arange(120, dtype=np.uint16).reshape(10, 12) * 500).astype(np.uint16)
    path = tmp_path / "gray16.png"
    assert imwrite(str(path), img)

    np.testing.assert_array_equal(imread(str(path), IMREAD_UNCHANGED), img)
    np.testing.assert_array_equal(imread(str(path), IMREAD_GRAYSCALE), (img >> 8).astype(np.uint8))


def test_imdecode_jpeg_bytes(tmp_path):
    img = np.full((16, 16, 3), (30, 90, 200), dtype=np.uint8)
    path = tmp_path / "flat.jpg"
    assert imwrite(str(path), img)

    decoded = imdecode(path.read_bytes(), IMREAD_UNCHANGED, path_hint="flat.jpg")
    assert decoded.shape == img.shape
    np.testing.assert_allclose(decoded, img, atol=3)
    gray = imdecode(path.read_bytes(), IMREAD_GRAYSCALE, path_hint="flat.jpg")
    assert gray.shape == (16, 16)


def test_imread_missing_or_invalid_file_returns_none(tmp_path):
    assert imread(str(tmp_path / "missing.png")) is None
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert imread(str(bad)) is None


# ---------------------------------------------------------------------------
# image_ops geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 3, 5])
def test_resize_area_halving_is_block_mean(channels):
    rng = np.random.default_rng(seed=channels)
    img = rng.integers(0, 256, size=(8, 12, channels)).astype(np.uint16) * 200
    out = image_ops.resize(img, (6, 4), interpolation="area")
    expected = img.reshape(4, 2, 6, 2, channels).astype(np.float64).mean(axis=(1, 3))
    assert out.shape == (4, 6, channels)
    assert out.dtype == img.dtype
    np.testing.assert_allclose(out, expected, atol=0.5)


def test_resize_linear_doubling_interpolates_half_pixel_centres():
    img = np.array([[0, 100]], dtype=np.uint8)
    out = image_ops.resize(img, (4, 1))
    np.testing.assert_array_equal(out, [[0, 25, 75, 100]])


@pytest.mark.parametrize("channels", [1, 3, 6])
def test_warp_affine_translation_fills_border_per_channel(channels):
    rng = np.random.default_rng(seed=channels)
    img = rng.integers(0, 256, size=(10, 12, channels), dtype=np.uint8)
    bv = tuple(range(1, channels + 1))
    matrix = np.array([[1.0, 0, 3], [0, 1, 2]])

    out = image_ops.warp_affine(img, matrix, (12, 10), bv)

    np.testing.assert_array_equal(out[2:, 3:], img[:-2, :-3])
    for c in range(channels):
        assert (out[:2, :, c] == bv[c]).all()
        assert (out[:, :3, c] == bv[c]).all()


def test_warp_perspective_identity():
    img = np.random.default_rng(0).integers(0, 256, size=(9, 7, 3), dtype=np.uint8)
    out = image_ops.warp_perspective(img, np.eye(3), (7, 9), (0, 0, 0))
    np.testing.assert_array_equal(out, img)


def test_hsv_round_trip():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    hsv = image_ops.bgr_to_hsv(img)
    assert hsv[..., 0].max() < 180
    np.testing.assert_array_equal(image_ops.bgr_to_hsv(np.array([[[0, 0, 255]]], np.uint8)), [[[0, 255, 255]]])
    np.testing.assert_array_equal(image_ops.bgr_to_hsv(np.array([[[255, 0, 0]]], np.uint8)), [[[120, 255, 255]]])
    back = image_ops.hsv_to_bgr(hsv)
    # 8-bit hue quantises to 2 degrees, so the round trip is only approximate.
    assert np.abs(back.astype(int) - img).max() <= 8


def test_polygon_mask_fills_inside_and_edges():
    mask = image_ops.polygon_mask((10, 10), [np.array([[2, 2], [7, 2], [7, 6], [2, 6]])])
    expected = np.zeros((10, 10), np.uint8)
    expected[2:7, 2:8] = 255
    np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# Tar-shard mode + multi-channel TIFFs
# ---------------------------------------------------------------------------


def _encode_multisample_tiff_bytes(img):
    """Serialize a (H, W, C) array as a single-page multi-sample TIFF."""
    buf = BytesIO()
    n = img.shape[-1] if img.ndim == 3 else 1
    if img.ndim == 3 and n not in (1, 3, 4):
        if n == 2:
            kwargs = dict(photometric="minisblack", extrasamples=["unassalpha"])
        else:
            kwargs = dict(
                photometric="rgb",
                extrasamples=["unassalpha"] * (n - 3),
            )
        tifffile.imwrite(buf, img, **kwargs)
    else:
        tifffile.imwrite(buf, img)
    return buf.getvalue()


def _make_multichannel_tar_shard(shard_dir, num_samples=2, channels=5, hw=(48, 64)):
    """Create a ``shard_0.tar`` containing N multi-channel TIFFs + labels.

    Returns the list of (image_name, label_name) pairs written into the tar.
    Each label is one centered 20%-sized object of class 0.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    tar_path = shard_dir / "shard_0.tar"
    h, w = hw
    members = []
    with tarfile.open(tar_path, "w") as tf:
        for i in range(num_samples):
            if channels == 1:
                img = np.full((h, w), 128, dtype=np.uint8)
            else:
                img = np.zeros((h, w, channels), dtype=np.uint8)
                for c in range(channels):
                    # Distinct per-channel + per-sample fill so any channel
                    # mix-up would corrupt the read-back values.
                    img[..., c] = (i + 1) * 7 + c * 11

            img_name = f"img_{i:03d}.tif"
            label_name = f"img_{i:03d}.txt"
            members.append((img_name, label_name))

            img_bytes = _encode_multisample_tiff_bytes(img)
            info = tarfile.TarInfo(name=img_name)
            info.size = len(img_bytes)
            tf.addfile(info, BytesIO(img_bytes))

            label_bytes = b"0 0.5 0.5 0.2 0.2\n"
            info = tarfile.TarInfo(name=label_name)
            info.size = len(label_bytes)
            tf.addfile(info, BytesIO(label_bytes))

    return tar_path, members


def test_tar_shard_detection_recognizes_tif_only_shard(tmp_path):
    shard_dir = tmp_path / "subset"
    _make_multichannel_tar_shard(shard_dir, num_samples=1, channels=5)

    assert is_tar_shard_yolo_subset_dir(shard_dir)

    samples = build_yolo_tar_shard_samples(shard_dir)
    assert len(samples) == 1
    tar_path, img_member, _, label_member = samples[0]
    assert img_member == "img_000.tif"
    assert label_member == "img_000.txt"
    assert tar_path.name == "shard_0.tar"


@pytest.mark.parametrize("channels", [2, 3, 4, 5])
def test_tar_shard_tiff_size_reads_through_bytes(tmp_path, channels):
    """The label-cache path inside ``cache_labels`` reads TIFF size from a
    ``BytesIO`` of bytes extracted from the tar. Exercise that hop directly
    so any regression in ``imdecode``/``tiff_size`` for multi-channel TIFFs
    surfaces here instead of as a multi-minute training subprocess failure.
    """
    shard_dir = tmp_path / "subset"
    h, w = 36, 52
    _make_multichannel_tar_shard(
        shard_dir, num_samples=1, channels=channels, hw=(h, w)
    )

    samples = build_yolo_tar_shard_samples(shard_dir)
    tar_path, img_member, _, _ = samples[0]
    raw = read_tar_member_bytes(tar_path, img_member)

    assert tiff_size(BytesIO(raw)) == (w, h)


@pytest.mark.parametrize("channels", [2, 3, 4, 5])
def test_load_images_and_labels_tar_shard_multichannel(tmp_path, channels):
    """End-to-end (sans training) check that the dataset loader produces
    correctly-shaped tensors when reading multi-channel TIFFs out of a tar
    shard. Covers the full path: tar enumeration, label cache (which itself
    goes through ``tiff_size`` on a ``BytesIO``), ``load_image``'s
    ``imdecode`` branch with ``IMREAD_UNCHANGED``, the channel-count
    assertion, ``letterbox`` (incl. ``_pad_constant`` for 5+ channels),
    and the final HWC→CHW transpose.
    """
    shard_dir = tmp_path / "subset"
    h, w = 48, 64
    num_samples = 2
    _make_multichannel_tar_shard(
        shard_dir, num_samples=num_samples, channels=channels, hw=(h, w)
    )

    dataset = LoadImagesAndLabels(
        path=str(shard_dir),
        img_size=128,
        batch_size=1,
        augment=False,
        hyp=None,
        rect=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
        num_channels=channels,
    )

    assert dataset.tar_shard_mode is True
    assert len(dataset) == num_samples
    # File refs should round-trip through the tar member helper.
    tar_path, img_member = split_tar_member_ref(dataset.img_files[0])
    assert tar_path.name == "shard_0.tar"
    assert img_member.endswith(".tif")
    # cache_labels recorded the logical (W, H) of the TIFF.
    np.testing.assert_array_equal(dataset.shapes[0], np.array([w, h]))

    img_tensor, labels_out, path, shapes = dataset[0]
    assert img_tensor.shape == (channels, 128, 128)
    assert img_tensor.dtype.is_floating_point is False  # uint8 tensor
    # One label row prepended with the batch index slot.
    assert labels_out.shape == (1, 6)


def test_load_images_and_labels_tar_shard_channel_mismatch_raises(tmp_path):
    """If a user passes ``num_channels`` that disagrees with the TIFF on disk,
    ``load_image`` should assert loudly rather than silently producing wrong
    tensors.
    """
    shard_dir = tmp_path / "subset"
    _make_multichannel_tar_shard(shard_dir, num_samples=1, channels=5)

    dataset = LoadImagesAndLabels(
        path=str(shard_dir),
        img_size=64,
        batch_size=1,
        augment=False,
        hyp=None,
        rect=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix="",
        num_channels=3,
    )

    with pytest.raises(AssertionError, match="num_channels"):
        dataset[0]
