#!/usr/bin/python
# -*- encoding: utf-8 -*-
import torch
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from .config import config
from . import net

class Solver_makeupGAN(object):
    def __init__(self):
        # Model hyper-parameters
        self.g_conv_dim = 64
        self.d_conv_dim = 64
        self.g_repeat_num = 6
        self.d_repeat_num = 3

        # add by VBT
        self.stage = [False, True, False]

        self.build_model()

    def to_var(self, x, requires_grad=True):
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad).float()
        else:
            return Variable(x).float()

    def de_norm(self, x):
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def build_model(self):
        self.G = net.Generator_spade(self.g_conv_dim, self.g_repeat_num, stage=self.stage, use_atten=True, use_diff=True)
        self.G.load_state_dict(torch.load(config.pwd + '/G.pth', map_location=torch.device('cpu')))
        self.G.eval()

    def rebound_box(self, mask_A, mask_B, mask_A_face, n1=10):
        """A: left      B: right"""
        index_tmp = mask_A.nonzero()
        x_A_index = index_tmp[:, 2]
        y_A_index = index_tmp[:, 3]
        index_tmp = mask_B.nonzero()
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        mask_A_temp = mask_A.copy_(mask_A)
        mask_B_temp = mask_B.copy_(mask_B)

        n2 = n1 + 1
        mask_A_temp[:, :, min(x_A_index) - n1:max(x_A_index) + n2, min(y_A_index) - n1:max(y_A_index) + n2] = \
                mask_A_face[:, :, min(x_A_index) - n1:max(x_A_index) + n2, min(y_A_index) - n1:max(y_A_index) + n2]

        mask_B_temp[:, :, min(x_B_index) - n1:max(x_B_index) + n2, min(y_B_index) - n1:max(y_B_index) + n2] = \
                mask_A_face[:, :, min(x_B_index) - n1:max(x_B_index) + n2, min(y_B_index) - n1:max(y_B_index) + n2]

        mask_A_temp = self.to_var(mask_A_temp, requires_grad=False)
        mask_B_temp = self.to_var(mask_B_temp, requires_grad=False)
        return mask_A_temp, mask_B_temp

    def generate(self, org_A, ref_B, lms_A=None, lms_B=None, mask_list_A=None, mask_list_B=None, diff_A=None,
                 diff_B=None, gamma=None, beta=None, ret=False):
        """org_A is content, ref_B is style"""
        res = self.G.forward_atten(org_A, ref_B, mask_list_A, mask_list_B, diff_A, diff_B, gamma, beta, ret)
        return res

    # mask attribute: 0:background 1:face 2:left-eyebrown 3:right-eyebrown 4:left-eye 5: right-eye 6: nose
    # 7: upper-lip 8: teeth 9: under-lip 10:hair 11: left-ear 12: right-ear 13: neck

    def test(self, img_A, mask_A, diff_A, img_B, mask_B, diff_B):
        cur_prama = None
        with torch.no_grad():
            real_org = self.to_var(img_A.unsqueeze(0))
            real_ref = self.to_var(img_B.unsqueeze(0))   # (b, c, h, w)
            mask_A = self.to_var(mask_A.unsqueeze(0), requires_grad=False)
            mask_B = self.to_var(mask_B.unsqueeze(0), requires_grad=False)
            diff_A = self.to_var(torch.Tensor(diff_A).unsqueeze(0), requires_grad=False)
            diff_B = self.to_var(torch.Tensor(diff_B).unsqueeze(0), requires_grad=False)

            mask_list_A, mask_list_B = [], []
            mask_A_lip = (mask_A == 7).float() + (mask_A == 9).float()
            mask_B_lip = (mask_B == 7).float() + (mask_B == 9).float()
            mask_list_A.append(mask_A_lip)
            mask_list_B.append(mask_B_lip)

            mask_A_eye_left = (mask_A == 4).float()
            mask_A_eye_right = (mask_A == 5).float()
            mask_B_eye_left = (mask_B == 4).float()
            mask_B_eye_right = (mask_B == 5).float()
            mask_A_face = (mask_A == 1).float() + (mask_A == 6).float()
            mask_B_face = (mask_B == 1).float() + (mask_B == 6).float()

            mask_list_A.append(mask_A_face)
            mask_list_B.append(mask_B_face)

            # avoid the situation that images with eye closed
            no_eyes = False
            if not ((mask_A_eye_left > 0).any() and (mask_B_eye_left > 0).any() and
                    (mask_A_eye_right > 0).any() and (mask_B_eye_right > 0).any()):
                no_eyes = True

            mask_A_eye_left_ori, mask_A_eye_right_ori = mask_A_eye_left, mask_A_eye_right
            mask_B_eye_left_ori, mask_B_eye_right_ori = mask_B_eye_left, mask_B_eye_right

            if not no_eyes:
                mask_A_eye_left, mask_A_eye_right = self.rebound_box(mask_A_eye_left_ori, mask_A_eye_right_ori, mask_A_face, n1=16)
                mask_B_eye_left, mask_B_eye_right = self.rebound_box(mask_B_eye_left_ori, mask_B_eye_right_ori, mask_B_face, n1=16)
                mask_list_A.append(mask_A_eye_left + mask_A_eye_right)
                mask_list_B.append(mask_B_eye_left + mask_B_eye_right)
            else:
                mask_list_A.append(torch.zeros_like(mask_list_A[0]))
                mask_list_B.append(torch.zeros_like(mask_list_B[0]))

            # 为后面的attention预处理mask和diff
            mask_A_aug = torch.cat(mask_list_A, 0)      # (3, 1, h, w)
            mask_B_aug = torch.cat(mask_list_B, 0)
            diff_size = (config.diff_size, config.diff_size)
            diff_A_re = F.interpolate(diff_A, size=diff_size).repeat(3, 1, 1, 1)  # (3, 136, diff_size, diff_size)
            diff_B_re = F.interpolate(diff_B, size=diff_size).repeat(3, 1, 1, 1)
            mask_A_re = F.interpolate(mask_A_aug, size=diff_size).repeat(1, diff_A.shape[1], 1, 1)  # (3, 136, diff_size, diff_size)
            mask_B_re = F.interpolate(mask_B_aug, size=diff_size).repeat(1, diff_B.shape[1], 1, 1)
            print(diff_A_re.shape)
            print(mask_A_re.shape)
            diff_A_re = diff_A_re * mask_A_re
            diff_B_re = diff_B_re * mask_B_re  # (3, 136, 32, 32)

            # 先mask，再分部位norm
            norm_A = torch.norm(diff_A_re, dim=1, keepdim=True).repeat(1, diff_A_re.shape[1], 1, 1)
            norm_B = torch.norm(diff_B_re, dim=1, keepdim=True).repeat(1, diff_B_re.shape[1], 1, 1)
            norm_A = torch.where(norm_A == 0, torch.tensor(1e10), norm_A)
            norm_B = torch.where(norm_B == 0, torch.tensor(1e10), norm_B)
            diff_A_re /= norm_A
            diff_B_re /= norm_B

            if not no_eyes:
                cur_prama = self.generate(real_org, real_ref, None, None, mask_list_A, mask_list_B, 
                                          diff_A_re, diff_B_re, ret=True)
            fake_A = self.generate(real_org, real_ref, None, None, mask_list_A, mask_list_B, diff_A_re,
                                   diff_B_re, gamma=cur_prama[0], beta=cur_prama[1])
            fake_A = fake_A.squeeze(0)

            # normalize
            min_, max_ = fake_A.min(), fake_A.max()
            fake_A.add_(-min_).div_(max_ - min_ + 1e-5)

            return ToPILImage()(fake_A)

