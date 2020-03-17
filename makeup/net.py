#!/usr/bin/python
# -*- encoding: utf-8 -*-
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
    def __init__(self, dim, pnet):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=pnet),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim, affine=pnet),
        )

    def forward(self, x):
        return x + self.main(x)


class GetSPADE(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.get_gamma = nn.Conv2d(dim_in, 1, kernel_size=1, stride=1, padding=0, bias=False)
        self.get_beta = nn.Conv2d(dim_in, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return gamma, beta


def nonLocalBlock2D(source, weight):
    """return: 从source中采样，得到target的形状"""
    assert source.shape == (3, 1, 64, 64)
    g_source = source.view(3, 1, -1)  # (N, C, H*W)
    g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)
    y = [weight[i] @ g_source[i] for i in range(3)]
    y = y[0] + y[1] + y[2]
    y = y.transpose().reshape(1, 1, 64, 64)
    return torch.tensor(y)


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
        self.simple_spade = GetSPADE(curr_dim)

        for i in range(3):
            setattr(self, f'pnet_bottleneck_{i+1}', ResidualBlock(dim=curr_dim, pnet=True))

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
            setattr(self, f'tnet_bottleneck_{i+1}', ResidualBlock(dim=curr_dim, pnet=False))

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
    def atten_feature(mask_s, weight, gamma_s, beta_s):
        """
        feature size: (1, c, h, w)
        mask_c(s): (3, 1, h, w)
        diff_c: (1, 138, 256, 256)
        return: (1, c, h, w)
        """
        gamma_s = gamma_s.repeat(3, 1, 1, 1) * mask_s  # (3, c, h, w) 每个部位分别提取，变成batch
        beta_s = beta_s.repeat(3, 1, 1, 1) * mask_s

        gamma = nonLocalBlock2D(gamma_s, weight)  # (3, c, h, w)
        beta = nonLocalBlock2D(beta_s, weight)
        return gamma, beta

    @staticmethod
    def param(mask, fea, diff):
        """
        feature size: (1, 256, 64, 64)
        diff: (3, 136, 32, 32)
        """
        mask_re = mask.repeat(1, 256, 1, 1)  # (3, c, h, w)
        fea = fea.repeat(3, 1, 1, 1)                 # (3, c, h, w)
        fea = fea * mask_re                        # (3, c, h, w) 最后做atten的fea。3代表3个部位。
        _input = torch.cat((fea * 0.01, diff), dim=1)
        return _input.view(3, -1, 64 * 64)  # (N, C+136, H*W)

    @staticmethod
    def get_weight(theta, phi):

        def ones(mat):
            result = mat.copy()
            result.data = np.ones_like(result.data)
            return result

        theta = sp.csr_matrix(theta)
        phi = sp.csc_matrix(phi)
        weight = theta @ phi
        weight *= 200
        
        maximums = weight.max(axis=-1)
        maximums += ones(maximums) * 0.01 # max will be subtracted to -0.01, otherwise omitted in sp
        
        maximums = np.array(maximums.todense().squeeze())[0]
        maximums = sp.diags(maximums, format='coo')
        weight -= maximums @ ones(weight)
        weight.data = np.exp(weight.data)
        
        sums = np.array(weight.sum(axis=-1).squeeze())[0]
        sums = sp.diags(sums, format='coo')
        sums *= ones(weight)
        weight.data = np.divide(weight.data, sums.data)
        return weight

    def forward_atten(self, c, s, mask_c, mask_s, diff_c, diff_s, gamma=None, beta=None, ret=False):
        """attention version
        c: (b, c, h, w)
        mask_list_c: lip, skin, eye. (b, 1, h, w)
        """
        s = self.pnet_in(s)

        # forward c in tnet
        c = self.tnet_in_conv(c)
        c = self.tnet_in_spade(c)
        c = self.tnet_in_relu(c)

        # down-sampling
        for i in range(2):
            if gamma is None:
                s = getattr(self, f'pnet_down_{i+1}')(s)
            c = getattr(self, f'tnet_down_conv_{i+1}')(c)
            c = getattr(self, f'tnet_down_spade_{i+1}')(c)
            c = getattr(self, f'tnet_down_relu_{i+1}')(c)

        # bottleneck
        for i in range(6):
            if gamma is None and i <= 2:
                cur_pnet_bottleneck = getattr(self, f'pnet_bottleneck_{i+1}')
            cur_tnet_bottleneck = getattr(self, f'tnet_bottleneck_{i+1}')

            # get s_pnet from p and transform
            if i == 3:
                if gamma is None:
                    theta = self.param(mask_c, c, diff_c).permute(0, 2, 1)  # (N, H*W, C+136)
                    phi = self.param(mask_s, s, diff_s)
                    weight = [self.get_weight(theta[i], phi[i]) for i in range(3)]
                    gamma, beta = self.simple_spade(s)
                    gamma, beta = self.atten_feature(mask_s, weight, gamma, beta)
                    if ret:
                        return [gamma, beta]

                c = c * (1 + gamma) + beta

            if gamma is None and i <= 2:
                s = cur_pnet_bottleneck(s)
            c = cur_tnet_bottleneck(c)

        # up-sampling
        for i in range(2):
            c = getattr(self, f'tnet_up_conv_{i+1}')(c)
            c = getattr(self, f'tnet_up_spade_{i+1}')(c)
            c = getattr(self, f'tnet_up_relu_{i+1}')(c)

        c = self.tnet_out(c)
        return c

