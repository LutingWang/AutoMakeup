#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
from torchvision.transforms import ToPILImage

from .config import config
from . import net


class Solver_makeupGAN(object):

    def __init__(self):
        # add by VBT
        self.stage = [False, True, False]

        # build model
        self.G = net.Generator_spade(64, 6, stage=self.stage, use_atten=True, use_diff=True)
        self.G.load_state_dict(torch.load(config.pwd + '/G.pth', map_location=torch.device('cpu')))
        self.G.eval()

    def generate(self, org_A, ref_B, lms_A=None, lms_B=None, mask_list_A=None, mask_list_B=None, 
                 diff_A=None, diff_B=None, gamma=None, beta=None, ret=False):
        """org_A is content, ref_B is style"""
        res = self.G.forward_atten(org_A, ref_B, mask_list_A, mask_list_B, diff_A, diff_B, gamma, beta, ret)
        return res

    # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose
    # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck

    def test(self, real_A, mask_A, diff_A_re, real_B, mask_B, diff_B_re):
        cur_prama = None
        with torch.no_grad():
            cur_prama = self.generate(real_A, real_B, None, None, mask_A, mask_B, 
                                      diff_A_re, diff_B_re, ret=True)
            fake_A = self.generate(real_A, real_B, None, None, mask_A, mask_B, 
                                   diff_A_re, diff_B_re, gamma=cur_prama[0], beta=cur_prama[1])
        fake_A = fake_A.squeeze(0)

        # normalize
        min_, max_ = fake_A.min(), fake_A.max()
        fake_A.add_(-min_).div_(max_ - min_ + 1e-5)

        return ToPILImage()(fake_A)

