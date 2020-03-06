#!/usr/bin/python
# -*- encoding: utf-8 -*-
import sys

import numpy as np
from PIL import Image
import torch
from torch.backends import cudnn
from torchvision import transforms

from .config import config
sys.path.append(config.pwd + '/..')
import faceutils as futils
from .solver_makeup import Solver_makeupGAN

cudnn.benchmark = True
solver = Solver_makeupGAN()

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

_fix = np.zeros((256, 256, 68 * 2))
for i in range(256):  # 行 (y) h
    for j in range(256):  # 列 (x) w
        _fix[i, j, :68] = i  # 赋值y
        _fix[i, j, 68:] = j  # 赋值x
_fix = _fix.transpose((2, 0, 1))  # (138, h, w)


def _ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


def preprocess(image: Image):
    face = futils.dlib.detect(image)
    if not face:
        raise RuntimeException("no faces detected")
    face = face[0]
    image, face = futils.dlib.crop(image, face)
    lms = futils.dlib.landmarks(image, face) * 256 / image.width
    lms = lms.round().transpose((1, 0)).reshape(-1, 1, 1)   # transpose to (y-x)
    lms = np.tile(lms, (1, 256, 256))  # (136, h, w)
    diff = _fix - lms

    image = image.resize((512, 512), Image.ANTIALIAS)
    mask = futils.mask.mask(image).resize((256, 256), Image.ANTIALIAS)
    image = image.resize((256, 256), Image.ANTIALIAS)
    return [_transform(image), _ToTensor(mask), diff]
