#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp

import numpy as np
from PIL import Image
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(osp.split(osp.realpath(__file__))[0] + '/lms.dat')


def detect(image: Image) -> 'faces':
    return detector(np.asarray(image), 1)


def crop(image: Image, face) -> (Image, 'face'):
    width, height = image.size
    size = max(face.width(), face.height()) * 1.3
    size = min(width, height, int(size)) // 2
    center = face.center()

    left = center.x - size
    right = center.x + size
    if left < 0:
        right -= left
        left = 0
    elif right > width:
        left -= right - width
        right = width

    top = center.y - size
    bottom = center.y + size
    if top < 0:
        bottom -= top
        top = 0
    elif bottom > height:
        top -= bottom - height
        bottom = height

    box = (left, top, right, bottom)
    image = image.crop(box)
    face = dlib.rectangle(
        face.left() - left, 
        face.top() - top,
        face.right() - left, 
        face.bottom() - top,
        )

    return image, face, box


def landmarks(image: Image, face):
    shape = predictor(np.asarray(image), face).parts()
    return np.array([[p.y, p.x] for p in shape])


