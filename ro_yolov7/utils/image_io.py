"""File image I/O: tifffile for .tif/.tiff, OpenCV for everything else."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import tifffile

_TIFF_EXTS = frozenset({".tif", ".tiff"})


def is_tiff_path(path) -> bool:
    return Path(str(path)).suffix.lower() in _TIFF_EXTS


def imread(path, flags: int = cv2.IMREAD_COLOR):
    if is_tiff_path(path):
        try:
            return tifffile.imread(str(path), key=0)
        except (OSError, ValueError, tifffile.TiffFileError):
            return None
    return cv2.imread(path, flags)


def imdecode(data, flags: int, path_hint: str = ""):
    if path_hint and is_tiff_path(path_hint):
        try:
            return tifffile.imread(BytesIO(bytes(data)), key=0)
        except (OSError, ValueError, tifffile.TiffFileError):
            return None
    arr = data if isinstance(data, np.ndarray) else np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, flags)


def imwrite(path, img) -> bool:
    if is_tiff_path(path):
        try:
            tifffile.imwrite(str(path), img)
            return True
        except (OSError, ValueError, TypeError, tifffile.TiffFileError):
            return False
    return cv2.imwrite(str(path), img)
