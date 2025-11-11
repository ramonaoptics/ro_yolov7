import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
import yaml
import argparse
import subprocess
import sys
import pytest

import ro_yolov7
from ro_yolov7.train import train
from ro_yolov7.utils.torch_utils import select_device


def test_train_importable():
    from ro_yolov7.train import train  # noqa


def test_test_importable():
    from ro_yolov7.test import test  # noqa


@pytest.fixture
def ml_dataset():
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

    yield dataset_dir

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_train_one_epoch(ml_dataset):
    dataset_dir = ml_dataset

    # Create data.yaml file
    data_yaml = dataset_dir / 'data.yaml'
    data_config = {
        'train': str(dataset_dir / 'train'),
        'val': str(dataset_dir / 'val'),
        'test': str(dataset_dir / 'test'),
        'nc': 1,  # number of classes
        'names': ['object']  # class names
    }
    with open(data_yaml, 'w') as f:
        yaml.dump(data_config, f)

    cfg_yaml = Path(ro_yolov7.__file__).parent / 'cfg' / 'training' / 'yolov7-tiny.yaml'
    hyp_yaml = Path(ro_yolov7.__file__).parent / 'data' / 'hyp.scratch.tiny.yaml'

    # Setup training options
    opt = argparse.Namespace(
        weights='',  # Train from scratch
        cfg=str(cfg_yaml),
        data=str(data_yaml),
        hyp=str(hyp_yaml),
        epochs=1,  # Only 1 epoch for testing
        batch_size=1,
        total_batch_size=1,
        img_size=[640, 640],
        rect=False,
        resume=False,
        nosave=True,  # Don't save checkpoints
        notest=False,
        noautoanchor=True,  # Skip autoanchor check
        evolve=False,
        bucket='',
        cache_images=False,
        image_weights=False,
        device='cpu',  # Use CPU for testing
        multi_scale=False,
        single_cls=True,
        adam=False,
        sync_bn=False,
        workers=0,  # No multiprocessing for testing
        project=str(dataset_dir / 'runs'),
        entity=None,
        name='test',
        exist_ok=True,
        quad=False,
        linear_lr=False,
        label_smoothing=0.0,
        upload_dataset=False,
        bbox_interval=-1,
        save_period=-1,
        artifact_alias='latest',
        freeze=[0],
        v5_metric=False,
        global_rank=-1,
        local_rank=-1,
        world_size=1,
        save_dir=str(dataset_dir / 'runs' / 'test')
    )

    with open(hyp_yaml) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    device = select_device(opt.device, batch_size=opt.batch_size)

    # Run training for 1 epoch
    results = train(hyp, opt, device, tb_writer=None)

    # Basic assertion to ensure training completed
    assert results is not None, "Training should return results"


def test_training_from_subprocess(ml_dataset):
    dataset_dir = ml_dataset

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

    cfg_path = Path(ro_yolov7.__file__).parent / 'cfg' / 'training' / 'yolov7-tiny.yaml'
    hyp_path = Path(ro_yolov7.__file__).parent / 'data' / 'hyp.scratch.tiny.yaml'
    train_script = Path(ro_yolov7.__file__).parent / 'train.py'

    cmd = [
        'python',
        str(train_script),
        '--weights', '',  # Empty string to train from scratch
        '--cfg', str(cfg_path),
        '--data', str(data_yaml),
        '--hyp', str(hyp_path),
        '--epochs', '1',  # Just 1 epoch for testing
        '--batch-size', '1',
        '--img-size', '640', '640',
        '--device', 'cpu',  # Use CPU for testing
        '--workers', '0',
        '--name', 'subprocess_test',
        '--project', str(dataset_dir / 'runs'),
        '--exist-ok',
        '--single-cls'  # Single class training
    ]

    # Run the training command via subprocess
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300  # 5 minute timeout
    )

    # Check that training completed successfully
    assert result.returncode == 0, f"Training failed with return code {result.returncode}\nStderr: {result.stderr}"

    # Check that expected output files were created
    weights_dir = dataset_dir / 'runs' / 'subprocess_test' / 'weights'
    assert weights_dir.exists(), "Weights directory was not created"

    # Check for last.pt (should exist even with nosave=False by default)
    last_pt = weights_dir / 'last.pt'
    best_pt = weights_dir / 'best.pt'
    assert last_pt.exists() or best_pt.exists(), "No model weights were saved"
