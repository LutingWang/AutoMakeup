#!/usr/bin/python
# -*- encoding: utf-8 -*-
import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = False
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=use_affine)
        )

    def forward(self, x):
        return x + self.main(x)


class GetSPADE(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GetSPADE, self).__init__()
        # 这里统一把卷积核改成1*1
        self.get_gamma = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.get_beta = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return gamma, beta


class NONLocalBlock2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Conv2d(in_channels=1, out_channels=1,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, source, weight):
        """(b, c, h, w)
        src_diff: (3, 136, 32, 32)
        return: 从source中采样，得到target的形状
        """
        batch_size = source.size(0)

        g_source = source.view(batch_size, 1, -1)  # (N, C, H*W)
        g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)

        # y = torch.bmm(weight.to_dense(), g_source)
        y = [weight[i] @ g_source[i] for i in range(3)]
        y = torch.cat([torch.tensor(e).unsqueeze(0) for e in y])
        y = y.permute(0, 2, 1).contiguous()  # (N, C, H*W)
        y = y.view(batch_size, 1, *source.size()[2:])     # 为啥有*
        return y


class Generator_spade(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self):
        super().__init__()

        # ------------------------------------ PNet ---------------------------------------

        layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
        )
        self.pnet_in = layers

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            layers = nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True),
            )

            setattr(self, f'pnet_down_{i+1}', layers)
            curr_dim = curr_dim * 2

        # Bottleneck. All bottlenecks share the same attention module
        self.atten_bottleneck_g = NONLocalBlock2D()
        self.atten_bottleneck_b = NONLocalBlock2D()
        self.simple_spade = GetSPADE(curr_dim, 1)

        for i in range(3):
            setattr(self, f'pnet_bottleneck_{i+1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='p'))

        # --------------------------------------- TNet ---------------------------------------

        self.tnet_in_conv = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.tnet_in_spade = nn.InstanceNorm2d(64, affine=False)
        self.tnet_in_relu = nn.ReLU(inplace=True)

        # Down-Sampling
        curr_dim = 64
        for i in range(2):
            setattr(self, f'tnet_down_conv_{i+1}', nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
            setattr(self, f'tnet_down_spade_{i+1}', nn.InstanceNorm2d(curr_dim * 2, affine=False))

            setattr(self, f'tnet_down_relu_{i+1}', nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(6):
            setattr(self, f'tnet_bottleneck_{i+1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t'))

        # Up-Sampling
        for i in range(2):
            setattr(self, f'tnet_up_conv_{i+1}', nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            setattr(self, f'tnet_up_spade_{i+1}', nn.InstanceNorm2d(curr_dim // 2, affine=False))
            setattr(self, f'tnet_up_relu_{i+1}', nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers = nn.Sequential(
            nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
            nn.Tanh()
        )
        self.tnet_out = layers

    @staticmethod
    def atten_feature(mask_s, weight, gamma_s, beta_s, atten_module_g, atten_module_b):
        """
        feature size: (1, c, h, w)
        mask_c(s): (3, 1, h, w)
        diff_c: (1, 138, 256, 256)
        return: (1, c, h, w)
        """
        channel_num = gamma_s.shape[1]

        mask_s_re = F.interpolate(mask_s, size=gamma_s.shape[2:]).repeat(1, channel_num, 1, 1)
        gamma_s_re = gamma_s.repeat(3, 1, 1, 1)
        gamma_s = gamma_s_re * mask_s_re  # (3, c, h, w) 每个部位分别提取，变成batch
        beta_s_re = beta_s.repeat(3, 1, 1, 1)
        beta_s = beta_s_re * mask_s_re

        gamma = atten_module_g(gamma_s, weight)  # (3, c, h, w)
        beta = atten_module_b(beta_s, weight)

        gamma = (gamma[0] + gamma[1] + gamma[2]).unsqueeze(0)  # (c, h, w) 把三个部位合并
        beta = (beta[0] + beta[1] + beta[2]).unsqueeze(0)
        return gamma, beta

    @staticmethod
    def get_weight(mask_c, mask_s, fea_c, fea_s, diff_c, diff_s):
        """  s --> source; c --> target
        feature size: (1, 256, 64, 64)
        diff: (3, 136, 32, 32)
        """
        HW = 64 * 64
        batch_size = 3
        assert fea_s is not None   # fea_s when i==3
        # get 3 part fea using mask
        channel_num = fea_s.shape[1]

        mask_c_re = F.interpolate(mask_c, size=64).repeat(1, channel_num, 1, 1)  # (3, c, h, w)
        fea_c = fea_c.repeat(3, 1, 1, 1)                 # (3, c, h, w)
        fea_c = fea_c * mask_c_re                        # (3, c, h, w) 最后做atten的fea。3代表3个部位。

        mask_s_re = F.interpolate(mask_s, size=64).repeat(1, channel_num, 1, 1)
        fea_s = fea_s.repeat(3, 1, 1, 1)
        fea_s = fea_s * mask_s_re

        theta_input = torch.cat((fea_c * 0.01, diff_c), dim=1)
        phi_input = torch.cat((fea_s * 0.01, diff_s), dim=1)

        theta_target = theta_input.view(batch_size, -1, HW)  # (N, C+136, H*W)
        theta_target = theta_target.permute(0, 2, 1)  # (N, H*W, C+136)

        phi_source = phi_input.view(batch_size, -1, HW)  # (N, C+136, H*W)


        def get_weight_ch(theta, phi):
            theta = sp.csr_matrix(theta)
            phi = sp.csc_matrix(phi)
            weight = theta @ phi
            weight *= 200
        
            maximums = weight.max(axis=-1)
            ones = maximums.copy()
            ones.data = np.ones_like(ones.data)
            maximums += ones * 0.01 # max will be subtracted to -0.01, otherwise omitted in sp
        
            maximums = np.array(maximums.todense().squeeze())[0]
            maximums = sp.diags(maximums, format='coo')
            ones = weight.copy()
            ones.data = np.ones_like(ones.data)
            weight -= maximums @ ones
            weight.data = np.exp(weight.data)
        
            sums = np.array(weight.sum(axis=-1).squeeze())[0]
            sums = sp.diags(sums, format='coo')
            ones = weight.copy()
            ones.data = np.ones_like(ones.data)
            sums *= ones
            weight.data = np.divide(weight.data, sums.data)
            return weight
        
        weight = [get_weight_ch(theta_target[i], phi_source[i]) for i in range(3)]
        return weight


    def forward_atten(self, c, s, mask_c, mask_s, diff_c, diff_s, gamma=None, beta=None, ret=False):
        """attention version
        c: (b, c, h, w)
        mask_list_c: lip, skin, eye. (b, 1, h, w)
        """
        img_c = c
        img_s = s
        # forward c in tnet
        c_tnet = self.tnet_in_conv(c)
        s = self.pnet_in(s)
        c_tnet = self.tnet_in_spade(c_tnet)
        c_tnet = self.tnet_in_relu(c_tnet)

        # down-sampling
        for i in range(2):
            if gamma is None:
                cur_pnet_down = getattr(self, f'pnet_down_{i+1}')
                s = cur_pnet_down(s)

            cur_tnet_down_conv = getattr(self, f'tnet_down_conv_{i+1}')
            cur_tnet_down_spade = getattr(self, f'tnet_down_spade_{i+1}')
            cur_tnet_down_relu = getattr(self, f'tnet_down_relu_{i+1}')
            c_tnet = cur_tnet_down_conv(c_tnet)
            c_tnet = cur_tnet_down_spade(c_tnet)
            c_tnet = cur_tnet_down_relu(c_tnet)

        # bottleneck
        for i in range(6):
            if gamma is None and i <= 2:
                cur_pnet_bottleneck = getattr(self, f'pnet_bottleneck_{i+1}')
            cur_tnet_bottleneck = getattr(self, f'tnet_bottleneck_{i+1}')

            # get s_pnet from p and transform
            if i == 3:
                if gamma is None:               # not in test_mix
                    gamma, beta = self.simple_spade(s)
                    # if (mask_s[1] != mask_c[1]).any():  # 如果不是同一张图才atten
                    weight = self.get_weight(mask_c, mask_s, c_tnet, s, diff_c, diff_s)

                    gamma, beta = self.atten_feature(mask_s, weight, gamma, beta, self.atten_bottleneck_g, self.atten_bottleneck_b)
                    if ret:
                        return [gamma, beta]

                c_tnet = c_tnet * (1 + gamma) + beta

            if gamma is None and i <= 2:
                s = cur_pnet_bottleneck(s)
            c_tnet = cur_tnet_bottleneck(c_tnet)

        # up-sampling
        for i in range(2):
            cur_tnet_up_conv = getattr(self, f'tnet_up_conv_{i+1}')
            cur_tnet_up_spade = getattr(self, f'tnet_up_spade_{i+1}')
            cur_tnet_up_relu = getattr(self, f'tnet_up_relu_{i+1}')
            c_tnet = cur_tnet_up_conv(c_tnet)
            c_tnet = cur_tnet_up_spade(c_tnet)
            c_tnet = cur_tnet_up_relu(c_tnet)

        c_tnet = self.tnet_out(c_tnet)
        return c_tnet

