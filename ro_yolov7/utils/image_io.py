"""File image I/O: tifffile for .tif/.tiff, OpenCV for everything else."""

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


def tiff_size(path_or_stream) -> tuple[int, int]:
    """Return (width, height) of a TIFF's logical image without decoding pixels.

    PIL mis-reads non-standard multi-sample TIFFs (e.g. 2- or 5-channel
    micro-imagery), and tifffile writes such arrays as a multi-page stack when
    it cannot infer a photometric interpretation from shape alone. Use the
    series-level shape and axes so the returned size reflects the logical
    (H, W) regardless of how the file was serialized.
    """
    with tifffile.TiffFile(path_or_stream) as tf:
        series = tf.series[0]
        shape = series.shape
        axes = series.axes
        if "Y" in axes and "X" in axes:
            h = int(shape[axes.index("Y")])
            w = int(shape[axes.index("X")])
        else:
            page = tf.pages[0]
            w = int(page.imagewidth)
            h = int(page.imagelength)
        return w, h
