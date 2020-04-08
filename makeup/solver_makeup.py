#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os.path as osp
pwd = osp.split(osp.realpath(__file__))[0]

import torch
from torchvision.transforms import ToPILImage

from . import net


class Solver_makeupGAN(object):

    def __init__(self):
        self.G = net.Generator_spade()
        self.G.load_state_dict(
            torch.load(pwd + '/G.pth', map_location=torch.device('cpu')), 
            strict=False,
            )
        self.G.eval()

    # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose
    # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck

    def test(self, real_A, mask_A, diff_A, real_B, mask_B, diff_B):
        """A->src, B->ref"""
        fake_A = self.G.forward_atten(real_A, real_B, mask_A, mask_B, diff_A, diff_B)[0]

        # normalize
        min_, max_ = fake_A.min(), fake_A.max()
        fake_A.add_(-min_).div_(max_ - min_ + 1e-5)

        return ToPILImage()(fake_A)

