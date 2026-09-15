# -*- coding: utf-8 -*-
"""Torch-free glyph padding shared by runtime, training, and export."""

import numpy as np
from PIL import Image, ImageOps


def border_color(image):
    """Return the per-channel median color of an RGB image border."""

    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.size == 0:
        return 255, 255, 255
    border = np.concatenate(
        (
            array[0, :, :],
            array[-1, :, :],
            array[:, 0, :],
            array[:, -1, :],
        ),
        axis=0,
    )
    values = np.median(border, axis=0)
    return tuple(int(round(value)) for value in values)


def _output_size(size):
    if isinstance(size, int) and not isinstance(size, bool):
        dimensions = size, size
    elif isinstance(size, (tuple, list)) and len(size) == 2:
        dimensions = tuple(size)
    else:
        raise ValueError("size must be a positive integer or width/height pair")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in dimensions
    ):
        raise ValueError("size must contain positive integers")
    return dimensions


def pad_glyph_image(image, size=64):
    """Pad a glyph with its border median while preserving aspect ratio."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL Image")
    rgb_image = image.convert("RGB")
    if rgb_image.width <= 0 or rgb_image.height <= 0:
        raise ValueError("image dimensions must be positive")
    return ImageOps.pad(
        rgb_image,
        _output_size(size),
        method=Image.Resampling.LANCZOS,
        color=border_color(rgb_image),
        centering=(0.5, 0.5),
    )


class PadGlyphImage:
    """Pickle-safe transform wrapper around ``pad_glyph_image``."""

    def __init__(self, size=64):
        self.size = _output_size(size)

    def __call__(self, image):
        return pad_glyph_image(image, self.size)
