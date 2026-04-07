import tarfile
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from ro_yolov7.utils.datasets import (
    LoadImagesAndLabels,
    build_yolo_tar_shard_samples,
    is_tar_shard_yolo_subset_dir,
)


@pytest.fixture
def tiny_yolo_tar_shard_dir(tmp_path):
    subset = tmp_path / "training"
    subset.mkdir(parents=True)
    img = np.zeros((12, 12), dtype=np.uint8)
    img[2:10, 2:10] = 220
    buf = BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    img_bytes = buf.getvalue()
    label_bytes = b"0 0.5 0.5 0.5 0.5\n"
    with tarfile.open(subset / "shard_000000.tar", "w", format=tarfile.GNU_FORMAT) as tf:
        for name, data in [("sample.png", img_bytes), ("sample.txt", label_bytes)]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, BytesIO(data))
    return subset


def test_is_tar_shard_yolo_subset_dir(tiny_yolo_tar_shard_dir):
    assert is_tar_shard_yolo_subset_dir(tiny_yolo_tar_shard_dir)


def test_build_yolo_tar_shard_samples(tiny_yolo_tar_shard_dir):
    samples = build_yolo_tar_shard_samples(tiny_yolo_tar_shard_dir)
    assert len(samples) == 1
    ta, im, tb, lb = samples[0]
    assert im == "sample.png" and lb == "sample.txt"
    assert ta == tb


def test_load_images_and_labels_tar_shard(tiny_yolo_tar_shard_dir):
    ds = LoadImagesAndLabels(
        str(tiny_yolo_tar_shard_dir),
        img_size=64,
        batch_size=1,
        augment=False,
        hyp=None,
        rect=False,
    )
    assert len(ds) == 1
    img, labels, path, _ = ds[0]
    assert img.shape[0] == 1  # one channel
    assert labels.shape[0] >= 1
