import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
import tifffile
import yaml
import subprocess
import pytest

import ro_yolov7


def test_train_importable():
    from ro_yolov7.train import train  # noqa


def test_test_importable():
    from ro_yolov7.test import test  # noqa


@pytest.fixture
def ml_dataset():
    """Fixture to create a minimal dataset with data.yaml for testing"""
    temp_dir = tempfile.mkdtemp()
    dataset_dir = Path(temp_dir) / "test_dataset"

    # Create train, val, and test subdirectories
    for split in ['train', 'val', 'test']:
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # Create a blank grayscale image (matching the project's grayscale format)
        img = np.zeros((640, 640), dtype=np.uint8)
        img_path = split_dir / f'{split}_image.jpg'
        cv2.imwrite(str(img_path), img)

        # Create a corresponding label file with one annotation
        # Format: class x_center y_center width height (normalized 0-1)
        label_path = split_dir / f'{split}_image.txt'
        with open(label_path, 'w') as f:
            # Single object of class 0, centered, taking 20% of image
            f.write('0 0.5 0.5 0.2 0.2\n')

    # Create data.yaml file
    data_yaml = dataset_dir / 'data.yaml'
    data_config = {
        'train': str(dataset_dir / 'train'),
        'val': str(dataset_dir / 'val'),
        'test': str(dataset_dir / 'test'),
        'nc': 1,  # number of classes
        'names': ['testing']  # class names
    }
    with open(data_yaml, 'w') as f:
        yaml.dump(data_config, f)

    yield dataset_dir

    shutil.rmtree(temp_dir, ignore_errors=True)


def test_training_from_subprocess(ml_dataset):
    dataset_dir = ml_dataset
    data_yaml = dataset_dir / 'data.yaml'
    cfg_path = Path(ro_yolov7.__file__).parent / 'cfg' / 'training' / 'yolov7-tiny.yaml'
    hyp_path = Path(ro_yolov7.__file__).parent / 'data' / 'hyp.scratch.tiny.yaml'
    train_script = Path(ro_yolov7.__file__).parent / 'train.py'
    default_weights_path = Path(ro_yolov7.__file__).parent / 'yolov7-tiny.pt'

    cmd = [
        'python', str(train_script),
        '--weights', str(default_weights_path),
        '--cfg', str(cfg_path),
        '--data', str(data_yaml),
        '--hyp', str(hyp_path),
        '--epochs', '1',
        '--batch-size', '1',
        '--img-size', '640', '640',
        # we test on cpu here to ensure CI compatibility
        '--device', 'cpu',
        '--workers', '4',
        '--name', 'subprocess_test',
        '--project', str(dataset_dir / 'runs'),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )

    assert result.returncode == 0, \
        f"Training failed with return code {result.returncode}\nStderr: {result.stderr}"

    weights_dir = dataset_dir / 'runs' / 'subprocess_test' / 'weights'
    assert weights_dir.exists(), "Weights directory was not created"

    best_pt = weights_dir / 'best.pt'
    assert best_pt.exists(), "Best model weights were not saved correctly"


def _make_multichannel_dataset(num_channels, image_format=".tif"):
    """Create a minimal multichannel dataset with data.yaml for testing.

    Returns (temp_dir, dataset_dir). Caller is responsible for cleaning up
    ``temp_dir`` when finished.
    """
    temp_dir = Path(tempfile.mkdtemp())
    dataset_dir = temp_dir / "test_dataset"

    for split in ["train", "val", "test"]:
        split_dir = dataset_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        if num_channels == 1:
            img = np.full((640, 640), 128, dtype=np.uint8)
        else:
            img = np.zeros((640, 640, num_channels), dtype=np.uint8)
            # Populate each channel with a distinct non-trivial value so that
            # channel-order regressions (e.g. BGR swaps or alpha premultiplication
            # introduced by cv2 when loading TIFFs) will cause the pixel values
            # seen during training to differ from what was written to disk.
            for c in range(num_channels):
                img[..., c] = (c + 1) * 30

        img_path = split_dir / f"{split}_image{image_format}"
        if image_format in (".tif", ".tiff"):
            tifffile.imwrite(str(img_path), img, compression="zlib")
        else:
            cv2.imwrite(str(img_path), img)

        label_path = split_dir / f"{split}_image.txt"
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

    data_yaml = dataset_dir / "data.yaml"
    data_config = {
        "train": str(dataset_dir / "train"),
        "val": str(dataset_dir / "val"),
        "test": str(dataset_dir / "test"),
        "nc": 1,
        "names": ["testing"],
    }
    with open(data_yaml, "w") as f:
        yaml.dump(data_config, f)

    return temp_dir, dataset_dir


@pytest.fixture
def ml_dataset_multi_channel():
    """JPEG-based 3-channel dataset (legacy smoke test)."""
    temp_dir, dataset_dir = _make_multichannel_dataset(3, image_format=".jpg")
    try:
        yield dataset_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(params=[3, 4])
def ml_dataset_multi_channel_tif(request):
    """TIFF-based multi-channel datasets (the real-world training path)."""
    num_channels = request.param
    temp_dir, dataset_dir = _make_multichannel_dataset(
        num_channels, image_format=".tif"
    )
    try:
        yield dataset_dir, num_channels
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _run_training_subprocess(dataset_dir, num_channels):
    data_yaml = dataset_dir / "data.yaml"
    cfg_path = (
        Path(ro_yolov7.__file__).parent / "cfg" / "training" / "yolov7-tiny.yaml"
    )
    hyp_path = (
        Path(ro_yolov7.__file__).parent / "data" / "hyp.scratch.tiny.yaml"
    )
    train_script = Path(ro_yolov7.__file__).parent / "train.py"
    default_weights_path = Path(ro_yolov7.__file__).parent / "yolov7-tiny.pt"

    cmd = [
        "python",
        str(train_script),
        "--weights",
        str(default_weights_path),
        "--cfg",
        str(cfg_path),
        "--data",
        str(data_yaml),
        "--hyp",
        str(hyp_path),
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--img-size",
        "640",
        "640",
        "--num_channels",
        str(num_channels),
        "--device",
        "cpu",
        "--workers",
        "4",
        "--name",
        "subprocess_test",
        "--project",
        str(dataset_dir / "runs"),
    ]

    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


def test_training_from_subprocess_multi_channel(ml_dataset_multi_channel):
    dataset_dir = ml_dataset_multi_channel
    result = _run_training_subprocess(dataset_dir, num_channels=3)

    assert result.returncode == 0, (
        f"Training failed with return code {result.returncode}\n"
        f"Stderr: {result.stderr}"
    )

    weights_dir = dataset_dir / "runs" / "subprocess_test" / "weights"
    assert weights_dir.exists(), "Weights directory was not created"

    best_pt = weights_dir / "best.pt"
    assert best_pt.exists(), "Best model weights were not saved correctly"


def test_training_from_subprocess_multi_channel_tif(ml_dataset_multi_channel_tif):
    dataset_dir, num_channels = ml_dataset_multi_channel_tif
    result = _run_training_subprocess(dataset_dir, num_channels=num_channels)

    assert result.returncode == 0, (
        f"Training failed with return code {result.returncode}\n"
        f"Stderr: {result.stderr}"
    )

    weights_dir = dataset_dir / "runs" / "subprocess_test" / "weights"
    assert weights_dir.exists(), "Weights directory was not created"

    best_pt = weights_dir / "best.pt"
    assert best_pt.exists(), "Best model weights were not saved correctly"


def test_load_image_preserves_multi_channel_tiff_values():
    # Guard against regressions where multi-channel TIFFs saved by tifffile
    # (with ExtraSamples=UNASSALPHA or PhotometricInterpretation=RGB) are
    # corrupted by cv2.imread: cv2 silently swaps BGR channels and premultiplies
    # RGB by the "alpha" channel, which quietly destroys the training signal.
    from ro_yolov7.utils.datasets import _read_image_file, _decode_image_bytes

    for num_channels in (1, 3, 4):
        if num_channels == 1:
            ref = np.full((32, 32), 123, dtype=np.uint8)
        else:
            ref = np.zeros((32, 32, num_channels), dtype=np.uint8)
            for c in range(num_channels):
                ref[..., c] = (c + 1) * 30

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.tif"
            tifffile.imwrite(str(path), ref, compression="zlib")

            loaded = _read_image_file(path, num_channels=num_channels)
            if num_channels == 1:
                if loaded.ndim == 3:
                    loaded = loaded[..., 0]
                np.testing.assert_array_equal(loaded, ref)
            else:
                assert loaded.shape == ref.shape
                np.testing.assert_array_equal(loaded, ref)

            with open(path, "rb") as f:
                data = f.read()
            loaded_bytes = _decode_image_bytes(
                data, num_channels=num_channels, path=str(path)
            )
            if num_channels == 1 and loaded_bytes.ndim == 3:
                loaded_bytes = loaded_bytes[..., 0]
            np.testing.assert_array_equal(loaded_bytes, ref)
