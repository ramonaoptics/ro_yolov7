import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
import yaml
import argparse

import ro_yolov7
from ro_yolov7.train import train
from ro_yolov7.utils.torch_utils import select_device


def test_train_importable():
    from ro_yolov7.train import train  # noqa


def test_test_importable():
    from ro_yolov7.test import test  # noqa


def test_train_one_epoch():
    """Test training for 1 epoch with minimal dataset"""
    # Create temporary directory for the test
    temp_dir = tempfile.mkdtemp()

    try:
        # Create directory structure
        dataset_dir = Path(temp_dir) / "test_dataset"

        # Create train, val, and test subdirectories with images and labels folders
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
            'names': ['object']  # class names
        }
        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f)

        # Create minimal hyperparameters
        hyp_yaml = dataset_dir / 'hyp.yaml'
        hyp_config = {
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 0.0,  # No warmup for quick test
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 0.05,
            'cls': 0.5,
            'cls_pw': 1.0,
            'obj': 1.0,
            'obj_pw': 1.0,
            'iou_t': 0.20,
            'anchor_t': 4.0,
            'fl_gamma': 0.0,
            'hsv_h': 0.0,  # Disable augmentation for testing
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 0.0,  # Disable mosaic for testing
            'mixup': 0.0,
            'copy_paste': 0.0,
            'paste_in': 0.0,
            'loss_ota': 0  # Disable OTA loss for faster testing
        }
        with open(hyp_yaml, 'w') as f:
            yaml.dump(hyp_config, f)

        # Use existing yolov7-tiny.yaml config instead of creating a custom one
        cfg_path = Path(ro_yolov7.__file__).parent / 'cfg' / 'training' / 'yolov7-tiny.yaml'

        # Copy the config file to temp directory so we can use it
        cfg_yaml = dataset_dir / 'model.yaml'
        shutil.copy(str(cfg_path), str(cfg_yaml))

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

        # Create save directory
        Path(opt.save_dir).mkdir(parents=True, exist_ok=True)

        # Load hyperparameters
        with open(hyp_yaml) as f:
            hyp = yaml.load(f, Loader=yaml.SafeLoader)

        # Select device
        device = select_device(opt.device, batch_size=opt.batch_size)

        # Run training for 1 epoch
        results = train(hyp, opt, device, tb_writer=None)

        # Basic assertion to ensure training completed
        assert results is not None, "Training should return results"

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
