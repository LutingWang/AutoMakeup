#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np
from PIL import Image, ImageFilter

from . import faceplusplus as fpp
from . import dlibutils as dlib
from . import mask

_size = 200
_radius = 20
_values = list(range(1, 255, 255 // _radius))
_mask = np.ones((_size, _size)) * 255
for i in range(_radius):
    for j in range(i, _size - i): # top and bottom
        _mask[i, j] = _mask[_size - i - 1, j] = _values[i]
    for j in range(i + 1, _size - i - 1): # left and right
        _mask[j, i] = _mask[j, _size - i - 1] = _values[i]
_mask = Image.fromarray(_mask).convert('L')
del i, j


def merge(bg: Image, fg: Image, box: ('left', 'top', 'right', 'bottom'), radius=10) -> Image:
    bg = bg.filter(ImageFilter.GaussianBlur(radius=radius))
    size = box[2] - box[0]
    fg = fg.resize((size, size))
    mask = _mask.resize((size, size))
    bg.paste(fg, box=box, mask=mask)
    return bg

