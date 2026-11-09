"""Image operations on numpy arrays, in OpenCV's conventions.

The data pipeline was written against OpenCV: images are (H, W) or (H, W, C)
arrays, colour images are in BGR order, hue is 0..180 and pixel centres sit at
integer coordinates. These helpers keep those conventions, so the code that
calls them is unchanged, and reproduce OpenCV's arithmetic to within one grey
level unless a docstring says otherwise. They accept any number of channels.
"""

import numpy as np
import scipy.sparse
from PIL import Image, ImageDraw, ImageFont


def _round_to_dtype(x, dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        x = np.clip(np.rint(x), info.min, info.max)
    return x.astype(dtype)


def _resize_weights(n_in, n_out, area):
    # Source indices and weights of each output sample along one axis.
    scale = n_in / n_out
    if area and scale > 1:
        # Fraction of each source pixel inside the output pixel's footprint.
        lo = np.arange(n_out) * scale
        idx = np.floor(lo).astype(np.intp)[:, None] + np.arange(int(np.ceil(scale)) + 1)
        weights = np.minimum(lo[:, None] + scale, idx + 1) - np.maximum(lo[:, None], idx)
        weights = np.clip(weights, 0, None) / scale
    else:
        # Bilinear, half-pixel centres, edges replicated.
        src = (np.arange(n_out) + 0.5) * scale - 0.5
        i0 = np.floor(src).astype(np.intp)
        frac = src - i0
        idx = np.stack([i0, i0 + 1], axis=1)
        weights = np.stack([1 - frac, frac], axis=1)
    return np.clip(idx, 0, n_in - 1), weights.astype(np.float32)


def _resize_matrix(n_in, n_out, area):
    # Sparse (n_out, n_in) matrix of the weights; clipped duplicates add up.
    idx, weights = _resize_weights(n_in, n_out, area)
    rows = np.repeat(np.arange(n_out), idx.shape[1])
    return scipy.sparse.csr_matrix((weights.ravel(), (rows, idx.ravel())), shape=(n_out, n_in))


def resize(img, size, interpolation="linear"):
    """Resize ``img`` to ``size = (width, height)``, as ``cv2.resize`` does.

    ``interpolation`` is ``"linear"`` (``cv2.INTER_LINEAR``) or ``"area"``
    (``cv2.INTER_AREA``). Area averaging applies along an axis that shrinks;
    an axis that grows is interpolated linearly. Unlike ``cv2.resize``, a
    trailing channel axis of length 1 is kept.
    """
    width, height = size
    h, w = img.shape[:2]
    if (width, height) == (w, h):
        return img.copy()
    area = interpolation == "area"
    c = int(np.prod(img.shape[2:], dtype=np.intp))
    x = img.reshape(h, w * c)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32)
    if height != h:
        x = _resize_matrix(h, height, area) @ x
    if width != w:
        x = x.reshape(height, w, c).transpose(1, 0, 2).reshape(w, height * c)
        x = _resize_matrix(w, width, area) @ x
        x = x.reshape(width, height, c).transpose(1, 0, 2)
    return _round_to_dtype(x, img.dtype).reshape((height, width) + img.shape[2:])


def _warp(img, matrix, dsize, border_value, method):
    # ``matrix`` maps source to destination pixel coordinates, as in OpenCV.
    # Pillow samples the source at the inverse map, with pixel centres at
    # +0.5, and gives a fill colour to points outside the source rather than
    # blending them with it; a one pixel border of the fill colour around the
    # source makes the edge interpolate the way OpenCV's constant border does.
    to_pil = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]])
    pad = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]])
    inverse = to_pil @ pad @ np.linalg.inv(matrix) @ np.linalg.inv(to_pil)
    if method == Image.Transform.PERSPECTIVE:
        data = tuple((inverse / inverse[2, 2]).ravel()[:8])
    else:
        data = tuple(inverse[:2].ravel())

    planes = img[..., None] if img.ndim == 2 else img
    out = np.empty((dsize[1], dsize[0], planes.shape[2]), dtype=img.dtype)
    src = np.empty((planes.shape[0] + 2, planes.shape[1] + 2), dtype=np.float32)
    for c in range(planes.shape[2]):
        fill = float(border_value[c])
        src[...] = fill
        src[1:-1, 1:-1] = planes[..., c]
        warped = Image.fromarray(src).transform(
            dsize, method, data, resample=Image.Resampling.BILINEAR, fillcolor=fill
        )
        out[..., c] = _round_to_dtype(np.asarray(warped), img.dtype)
    return out[..., 0] if img.ndim == 2 else out


def warp_affine(img, matrix, dsize, border_value):
    """``cv2.warpAffine`` with bilinear interpolation and a constant border.

    ``matrix`` is the 2x3 (or 3x3) source to destination map, ``dsize`` the
    output ``(width, height)`` and ``border_value`` one value per channel.
    """
    matrix = np.vstack([np.asarray(matrix, dtype=np.float64)[:2], [0, 0, 1]])
    return _warp(img, matrix, dsize, border_value, Image.Transform.AFFINE)


def warp_perspective(img, matrix, dsize, border_value):
    """``cv2.warpPerspective`` with bilinear interpolation and a constant border."""
    return _warp(img, np.asarray(matrix, dtype=np.float64), dsize, border_value,
                 Image.Transform.PERSPECTIVE)


def rotation_matrix_2d(center, angle, scale):
    """The 2x3 matrix of ``cv2.getRotationMatrix2D``; ``angle`` in degrees."""
    a = scale * np.cos(np.deg2rad(angle))
    b = scale * np.sin(np.deg2rad(angle))
    cx, cy = center
    return np.array([[a, b, (1 - a) * cx - b * cy], [-b, a, b * cx + (1 - a) * cy]])


def bgr_to_hsv(img):
    """``cv2.COLOR_BGR2HSV`` for uint8: hue 0..180, saturation and value 0..255."""
    b, g, r = (img[..., i].astype(np.int32) for i in range(3))
    v = np.maximum(np.maximum(r, g), b)
    diff = v - np.minimum(np.minimum(r, g), b)
    # OpenCV divides through tables of (255 << 12) / v and (180 << 12) / (6 diff).
    shift = 12
    with np.errstate(divide="ignore"):
        sdiv = np.rint((255 << shift) / np.arange(256)).clip(0, None)
        hdiv = np.rint((180 << shift) / (6.0 * np.arange(256))).clip(0, None)
    sdiv[0] = hdiv[0] = 0
    sdiv, hdiv = sdiv.astype(np.int64), hdiv.astype(np.int64)
    s = (diff * sdiv[v] + (1 << (shift - 1))) >> shift
    h = np.where(v == r, g - b, np.where(v == g, b - r + 2 * diff, r - g + 4 * diff))
    h = (h * hdiv[diff] + (1 << (shift - 1))) >> shift
    h = np.where(h < 0, h + 180, h)
    return np.stack([h, s, v], axis=-1).astype(np.uint8)


def hsv_to_bgr(img):
    """``cv2.COLOR_HSV2BGR`` for uint8, the inverse of :func:`bgr_to_hsv`."""
    h = img[..., 0].astype(np.float32) * np.float32(6 / 180)
    s = img[..., 1].astype(np.float32) * np.float32(1 / 255)
    v = img[..., 2].astype(np.float32) * np.float32(1 / 255)
    h = np.where(h >= 6, h - 6, h)
    sector = np.floor(h).astype(np.intp)
    f = h - sector
    tab = np.stack([v, v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))])
    order = np.array([[1, 3, 0], [1, 0, 2], [3, 0, 1], [0, 2, 1], [0, 1, 3], [2, 1, 0]])
    bgr = np.stack([np.take_along_axis(tab, order[sector, i][None], 0)[0] for i in range(3)], -1)
    bgr = np.where((s == 0)[..., None], v[..., None], bgr)
    return _round_to_dtype(bgr * np.float32(255), np.uint8)


# Fixed point coefficients of OpenCV's 8 bit RGB <-> YUV conversions.
_YUV_SHIFT = 14
_R2Y, _G2Y, _B2Y, _R2V, _B2U = 4899, 9617, 1868, 14369, 8061
_U2B, _U2G, _V2G, _V2R = 33292, -6472, -9519, 18678


def rgb_to_yuv(img):
    """``cv2.COLOR_RGB2YUV`` for uint8; pass ``img[..., ::-1]`` for BGR."""
    r, g, b = (img[..., i].astype(np.int32) for i in range(3))
    half = 1 << (_YUV_SHIFT - 1)
    y = (r * _R2Y + g * _G2Y + b * _B2Y + half) >> _YUV_SHIFT
    delta = (128 << _YUV_SHIFT) + half
    u = ((b - y) * _B2U + delta) >> _YUV_SHIFT
    v = ((r - y) * _R2V + delta) >> _YUV_SHIFT
    return np.stack([y, u, v], axis=-1).clip(0, 255).astype(np.uint8)


def yuv_to_rgb(img):
    """``cv2.COLOR_YUV2RGB`` for uint8; reverse the result's channels for BGR."""
    y, u, v = (img[..., i].astype(np.int32) for i in range(3))
    u, v = u - 128, v - 128
    half = 1 << (_YUV_SHIFT - 1)
    r = y + ((v * _V2R + half) >> _YUV_SHIFT)
    g = y + ((u * _U2G + v * _V2G + half) >> _YUV_SHIFT)
    b = y + ((u * _U2B + half) >> _YUV_SHIFT)
    return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)


def equalize_hist(img):
    """``cv2.equalizeHist`` of a uint8 single channel image."""
    hist = np.bincount(img.ravel(), minlength=256)
    first = int(np.flatnonzero(hist)[0])
    if hist[first] == img.size:
        return np.full_like(img, first)
    scale = np.float32(255.0 / (img.size - hist[first]))
    lut = np.zeros(256, dtype=np.uint8)
    lut[first + 1:] = _round_to_dtype(np.cumsum(hist[first + 1:]).astype(np.float32) * scale, np.uint8)
    return lut[img]


def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """``cv2.createCLAHE(clip_limit, tile_grid_size).apply`` on a uint8 single channel image."""
    tiles_x, tiles_y = tile_grid_size
    h, w = img.shape
    src = img
    if h % tiles_y or w % tiles_x:
        # OpenCV reflects the image out to a whole number of tiles, and adds
        # a whole tile along an axis that already divides evenly.
        src = np.pad(img, ((0, tiles_y - h % tiles_y), (0, tiles_x - w % tiles_x)), mode="reflect")
    tile_h, tile_w = src.shape[0] // tiles_y, src.shape[1] // tiles_x
    tile_area = tile_h * tile_w
    limit = max(int(clip_limit * tile_area / 256), 1) if clip_limit > 0 else None

    tiles = src[: tiles_y * tile_h, : tiles_x * tile_w]
    tiles = tiles.reshape(tiles_y, tile_h, tiles_x, tile_w).transpose(0, 2, 1, 3)
    tiles = tiles.reshape(tiles_y * tiles_x, tile_area)
    offsets = np.arange(tiles.shape[0])[:, None] * 256
    hist = np.bincount((tiles + offsets).ravel(), minlength=tiles.shape[0] * 256)
    hist = hist.reshape(-1, 256)
    if limit is not None:
        clipped = np.clip(hist - limit, 0, None).sum(axis=1)
        hist = np.minimum(hist, limit) + (clipped // 256)[:, None]
        for t, residual in enumerate(clipped % 256):
            if residual:
                step = max(256 // residual, 1)
                hist[t, np.arange(0, 256, step)[:residual]] += 1
    lut_scale = np.float32(255.0 / tile_area)
    lut = _round_to_dtype(np.cumsum(hist, axis=1).astype(np.float32) * lut_scale, np.uint8)
    lut = lut.reshape(tiles_y, tiles_x, 256).astype(np.float32)

    def axis_weights(n, tile, count):
        pos = np.arange(n, dtype=np.float32) * np.float32(1.0 / tile) - np.float32(0.5)
        t1 = np.floor(pos).astype(np.intp)
        frac = pos - t1
        return np.clip(t1, 0, None), np.clip(t1 + 1, None, count - 1), frac

    y1, y2, fy = axis_weights(h, tile_h, tiles_y)
    x1, x2, fx = axis_weights(w, tile_w, tiles_x)
    y1, y2, fy = y1[:, None], y2[:, None], fy[:, None]
    top = lut[y1, x1, img] * (1 - fx) + lut[y1, x2, img] * fx
    bottom = lut[y2, x1, img] * (1 - fx) + lut[y2, x2, img] * fx
    return _round_to_dtype(top * (1 - fy) + bottom * fy, np.uint8)


def polygon_mask(shape, polygons):
    """A uint8 (H, W) mask, 255 inside ``polygons`` and on their edges, 0 elsewhere.

    Each polygon is an (N, 2) array of x, y pixel coordinates, as for
    ``cv2.drawContours(..., cv2.FILLED)``; edge pixels can differ from it.
    """
    mask = Image.new("L", (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        points = [tuple(p) for p in np.asarray(polygon, dtype=np.int32).reshape(-1, 2).tolist()]
        if len(points) >= 2:
            draw.polygon(points, fill=255, outline=255)
        elif points:
            draw.point(points, fill=255)
    return np.asarray(mask)


# Drawing. These draw into a uint8 array in place; a colour is one value per
# channel, in the array's own channel order. Lines are not antialiased.

def _channel_value(color, c):
    # OpenCV pads a colour with zeros to the image's channel count.
    color = np.atleast_1d(color)
    return int(color[c]) if c < len(color) else 0


def _draw(img, paint):
    if img.ndim == 3 and img.shape[2] == 3:
        canvas = Image.fromarray(img)
        paint(ImageDraw.Draw(canvas), lambda color: tuple(_channel_value(color, c) for c in range(3)))
        img[...] = np.asarray(canvas)
        return img
    planes = img[..., None] if img.ndim == 2 else img
    for c in range(planes.shape[2]):
        canvas = Image.fromarray(np.ascontiguousarray(planes[..., c]))
        paint(ImageDraw.Draw(canvas), lambda color, c=c: _channel_value(color, c))
        planes[..., c] = np.asarray(canvas)
    return img


def draw_rectangle(img, pt1, pt2, color, thickness=1):
    """``cv2.rectangle``: an outline ``thickness`` wide, centred on the edges, or filled if negative."""
    (x1, x2), (y1, y2) = sorted((pt1[0], pt2[0])), sorted((pt1[1], pt2[1]))

    def paint(draw, col):
        if thickness < 0:
            draw.rectangle([x1, y1, x2, y2], fill=col(color))
        else:
            o = thickness // 2
            draw.rectangle([x1 - o, y1 - o, x2 + o, y2 + o], outline=col(color), width=thickness)
    return _draw(img, paint)


def draw_line(img, pt1, pt2, color, thickness=1):
    """``cv2.line``."""
    return _draw(img, lambda draw, col: draw.line([tuple(pt1), tuple(pt2)], fill=col(color), width=thickness))


def draw_circle(img, center, radius, color, thickness=1):
    """``cv2.circle``: an outline, or a disc if ``thickness`` is negative."""
    x, y = center
    box = [x - radius, y - radius, x + radius, y + radius]

    def paint(draw, col):
        if thickness < 0:
            draw.ellipse(box, fill=col(color))
        else:
            draw.ellipse(box, outline=col(color), width=thickness)
    return _draw(img, paint)


def _font(font_scale):
    # Pillow's built in font at about the size of OpenCV's Hershey simplex.
    return ImageFont.load_default(size=max(30 * font_scale, 6))


def text_size(text, font_scale):
    """The ``(width, height)`` above the baseline of ``text``, as ``cv2.getTextSize(...)[0]``."""
    left, top, right, _ = _font(font_scale).getbbox(text, anchor="ls")
    return int(np.ceil(right - left)), int(np.ceil(-top))


def put_text(img, text, origin, font_scale, color):
    """``cv2.putText``: ``origin`` is the left end of the baseline."""
    font = _font(font_scale)
    return _draw(img, lambda draw, col: draw.text(tuple(origin), text, fill=col(color), font=font, anchor="ls"))
