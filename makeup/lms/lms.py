#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os.path as osp

import numpy as np
from PIL import Image
import dlib


pwd = osp.split(osp.realpath(__file__))[0]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(pwd + '/model.dat')

def gen(img):
    img = np.asarray(img)
    face = detector(img, 1)[0]
    shape = predictor(img, face).parts()
    return np.array([[p.y, p.x] for p in shape])
