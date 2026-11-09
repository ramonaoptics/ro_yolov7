"""Image and video I/O: tifffile for .tif/.tiff, Pillow for other images, imageio for video.

Images come back the way OpenCV's ``imread`` returns them, which is what the
rest of the package expects: colour channels in BGR(A) order, and the
``IMREAD_*`` flag choosing the channel count and depth.
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageOps

_TIFF_EXTS = frozenset({".tif", ".tiff"})

# Same values as OpenCV's flags.
IMREAD_UNCHANGED = -1
IMREAD_GRAYSCALE = 0
IMREAD_COLOR = 1

# Options OpenCV writes with by default where Pillow's differ.
_SAVE_OPTIONS = {".jpg": {"quality": 95}, ".jpeg": {"quality": 95}}


def is_tiff_path(path) -> bool:
    return Path(str(path)).suffix.lower() in _TIFF_EXTS


def _swap_red_blue(arr):
    # RGB(A) <-> BGR(A) as a new array; other channel counts are left alone.
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return arr[..., [2, 1, 0, 3][: arr.shape[2]]]
    return arr


def _to_uint8(arr):
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr >> 8).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _pil_to_array(im):
    if im.mode == "P":
        im = im.convert("RGBA" if "transparency" in im.info else "RGB")
    elif im.mode in ("1", "PA", "LA", "La", "RGBa", "CMYK", "YCbCr", "LAB", "HSV"):
        im = im.convert({"1": "L", "PA": "RGBA", "LA": "RGBA", "La": "RGBA", "RGBa": "RGBA"}.get(im.mode, "RGB"))
    arr = np.asarray(im)
    return arr.astype(arr.dtype.newbyteorder("=")) if arr.dtype.byteorder == ">" else arr


def _decode(source, flags):
    try:
        with Image.open(source) as im:
            if flags != IMREAD_UNCHANGED:
                if flags == IMREAD_GRAYSCALE and im.format == "JPEG":
                    # Decode straight to luma, as OpenCV has libjpeg do.
                    im.draft("L", None)
                im = ImageOps.exif_transpose(im)  # OpenCV applies the orientation too
            arr = _pil_to_array(im)
    except (OSError, ValueError, SyntaxError):
        return None

    if flags == IMREAD_UNCHANGED:
        arr = _swap_red_blue(arr)
    else:
        arr = _to_uint8(arr)
        if flags == IMREAD_GRAYSCALE and arr.ndim == 3:
            # cv2.COLOR_RGB2GRAY's fixed point weights
            r, g, b = (arr[..., i].astype(np.uint32) for i in range(3))
            arr = ((r * 4899 + g * 9617 + b * 1868 + 8192) >> 14).astype(np.uint8)
        elif flags != IMREAD_GRAYSCALE:
            arr = np.repeat(arr[:, :, None], 3, axis=2) if arr.ndim == 2 else arr[..., [2, 1, 0]]
    return arr if arr.flags.writeable else arr.copy()


def imread(path, flags: int = IMREAD_COLOR):
    if is_tiff_path(path):
        try:
            return tifffile.imread(str(path), key=0)
        except (OSError, ValueError, tifffile.TiffFileError):
            return None
    return _decode(str(path), flags)


def imdecode(data, flags: int, path_hint: str = ""):
    if path_hint and is_tiff_path(path_hint):
        try:
            return tifffile.imread(BytesIO(bytes(data)), key=0)
        except (OSError, ValueError, tifffile.TiffFileError):
            return None
    return _decode(BytesIO(bytes(data)), flags)


def imwrite(path, img) -> bool:
    if is_tiff_path(path):
        try:
            tifffile.imwrite(str(path), img)
            return True
        except (OSError, ValueError, TypeError, tifffile.TiffFileError):
            return False
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    try:
        Image.fromarray(np.ascontiguousarray(_swap_red_blue(img))).save(
            str(path), **_SAVE_OPTIONS.get(Path(str(path)).suffix.lower(), {})
        )
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


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


def _imageio_ffmpeg():
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg
    except ImportError as e:
        raise ImportError(
            "Reading and writing video needs imageio and its ffmpeg plugin: "
            "pip install imageio imageio-ffmpeg"
        ) from e
    return imageio, imageio_ffmpeg


class VideoReader:
    """The frames of a video file, stream URL or camera index, in BGR order.

    Covers what the loaders used of ``cv2.VideoCapture``: ``read``, ``grab``
    and ``retrieve``, ``release``, and the ``fps``, ``width``, ``height`` and
    ``frame_count`` of the source (``frame_count`` is 0 when it is unknown).
    """

    def __init__(self, source):
        imageio, imageio_ffmpeg = _imageio_ffmpeg()
        self._reader = None
        self._frame = None
        if isinstance(source, int):
            source = f"<video{source}>"
        if "://" in source:
            # imageio would download an http(s) URL as a file, so give the
            # URL to ffmpeg directly.
            self._frames = imageio_ffmpeg.read_frames(source)
            meta = next(self._frames)
            self._raw = True
        else:
            self._reader = imageio.get_reader(source, format="FFMPEG")
            meta = self._reader.get_meta_data()
            self._frames = iter(self._reader)
            self._raw = False
        self.width, self.height = (int(v) for v in meta["size"])
        fps = meta.get("fps")
        self.fps = float(fps) if fps and np.isfinite(fps) else 0.0
        duration = meta.get("duration")
        known = duration and np.isfinite(duration) and self.fps
        self.frame_count = round(self.fps * duration) if known else 0
        self._open = True

    def isOpened(self):
        return self._open

    def grab(self):
        try:
            frame = next(self._frames)
        except (StopIteration, RuntimeError, OSError):
            self._frame = None
            return False
        if self._raw:
            frame = np.frombuffer(frame, dtype=np.uint8).reshape(self.height, self.width, 3)
        self._frame = np.ascontiguousarray(frame[..., ::-1])
        return True

    def retrieve(self):
        return self._frame is not None, self._frame

    def read(self):
        self.grab()
        return self.retrieve()

    def release(self):
        if self._open:
            self._open = False
            if self._reader is not None:
                self._reader.close()
            else:
                self._frames.close()


class VideoWriter:
    """Write BGR frames to a video file through imageio's ffmpeg plugin (H.264 for .mp4)."""

    def __init__(self, path, fps):
        imageio, _ = _imageio_ffmpeg()
        # A block size of 2 keeps the frame size, which yuv420p needs even.
        self._writer = imageio.get_writer(str(path), format="FFMPEG", fps=fps or 30, macro_block_size=2)

    def write(self, frame):
        self._writer.append_data(np.ascontiguousarray(frame[..., ::-1]))

    def release(self):
        self._writer.close()
