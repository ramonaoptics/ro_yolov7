"""
YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

Implementation of paper: YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors
https://arxiv.org/abs/2207.02696
"""

import warnings

# Suppress warnings from onnxconvertercommon during ONNX export
warnings.filterwarnings('ignore', module='onnxconvertercommon.*')

__version__ = "0.2.0"
__author__ = "Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao"

from ro_yolov7.models.yolo import Model
from ro_yolov7.models.experimental import attempt_load

__all__ = ["Model", "attempt_load", "__version__"]
