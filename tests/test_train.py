import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
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
        '--device', 'cpu',
        '--workers', '0',
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

    last_pt = weights_dir / 'last.pt'
    best_pt = weights_dir / 'best.pt'
    assert last_pt.exists() and best_pt.exists(), "Model weights were not saved correctly"
