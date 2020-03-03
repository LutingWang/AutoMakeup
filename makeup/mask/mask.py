#!/usr/bin/python
# -*- encoding: utf-8 -*-

from .model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


mapper = [0, 1, 2, 3, 4, 5, 0, 11, 12, 0, 6, 8, 7, 9, 13, 0, 0, 10, 0]
n_classes = 19
save_pth = osp.split(osp.realpath(__file__))[0] + '/res/res.pth'

net = BiSeNet(n_classes=19)
net.load_state_dict(torch.load(save_pth, map_location='cpu'))
net.eval()
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def gen(img):
    image = img.resize((512, 512), Image.ANTIALIAS)
    with torch.no_grad():
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    for r in range(parsing.shape[0]):
        parsing[r] = [mapper[x] for x in parsing[r]]
    parsing = Image.fromarray(parsing.astype(np.uint8))
    return (image.resize((256, 256), Image.ANTIALIAS), 
            parsing.resize((256, 256), Image.ANTIALIAS))


if __name__ == "__main__":
    img = Image.open('./test.png')
    img, parsing = evaluate(img)

