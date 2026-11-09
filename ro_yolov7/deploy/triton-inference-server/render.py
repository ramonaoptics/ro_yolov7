import numpy as np

from math import sqrt

from ro_yolov7.utils.image_ops import draw_rectangle, put_text, text_size

_LINE_THICKNESS_SCALING = 500.0

np.random.seed(0)
RAND_COLORS = np.random.randint(50, 255, (64, 3), "int")  # used for class visu
RAND_COLORS[0] = [220, 220, 220]

def render_box(img, box, color=(200, 200, 200)):
    """
    Render a box. Calculates scaling and thickness automatically.
    :param img: image to render into
    :param box: (x1, y1, x2, y2) - box coordinates
    :param color: (b, g, r) - box color
    :return: updated image
    """
    x1, y1, x2, y2 = box
    thickness = int(
        round(
            (img.shape[0] * img.shape[1])
            / (_LINE_THICKNESS_SCALING * _LINE_THICKNESS_SCALING)
        )
    )
    thickness = max(1, thickness)
    img = draw_rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness=thickness
    )
    return img

def render_filled_box(img, box, color=(200, 200, 200)):
    """
    Render a box. Calculates scaling and thickness automatically.
    :param img: image to render into
    :param box: (x1, y1, x2, y2) - box coordinates
    :param color: (b, g, r) - box color
    :return: updated image
    """
    x1, y1, x2, y2 = box
    img = draw_rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness=-1
    )
    return img

_TEXT_SCALING = 520.0


def get_text_size(img, text, normalised_scaling=1.0):
    """
    Get calculated text size (as box width and height)
    :param img: image reference, used to determine appropriate text scaling
    :param text: text to display
    :param normalised_scaling: additional normalised scaling. Default 1.0.
    :return: (width, height) - width and height of text box
    """
    scaling = img.shape[0] / _TEXT_SCALING * normalised_scaling
    return text_size(text, scaling)


def render_text(img, text, pos, color=(200, 200, 200), normalised_scaling=1.0):
    """
    Render a text into the image. Calculates scaling and thickness automatically.
    :param img: image to render into
    :param text: text to display
    :param pos: (x, y) - upper left coordinates of render position
    :param color: (b, g, r) - text color
    :param normalised_scaling: additional normalised scaling. Default 1.0.
    :return: updated image
    """
    x, y = pos
    scaling = img.shape[0] / _TEXT_SCALING * normalised_scaling
    size = get_text_size(img, text, normalised_scaling)
    put_text(img, text, (int(x), int(y + size[1])), scaling, color)
    return img
