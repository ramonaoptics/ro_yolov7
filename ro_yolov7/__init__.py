"""
YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors

Implementation of paper: YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors
https://arxiv.org/abs/2207.02696
"""
__version__ = "0.4.4"
__author__ = "Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao, John Efromson, Mark Harfouche"

from ro_yolov7.models.yolo import Model
from ro_yolov7.models.experimental import attempt_load

__all__ = ["Model", "attempt_load", "__version__"]

import sys

if 'owl' in sys.modules:
    raise ImportError("YOLOv7 is incompatible with the 'owl' package.")
