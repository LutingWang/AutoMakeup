import torch
import torch.nn as nn
import torch.nn.functional as F

from makeup.config import config


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input

def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out, net_mode=None):
        if net_mode == 'p' or (net_mode is None):
            use_affine = True
        elif net_mode == 't':
            use_affine = not config.tnet_whiten
        super(ResidualBlock, self).__init__()
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
        self.get_gamma = nn.Conv2d(dim_in, dim_out, kernel_size=config.k_size, stride=1, padding=1 if config.k_size==3 else 0, bias=False)
        self.get_beta = nn.Conv2d(dim_in, dim_out, kernel_size=config.k_size, stride=1, padding=1 if config.k_size==3 else 0, bias=False)
        # self.get_beta = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        gamma = self.get_gamma(x)
        beta = self.get_beta(x)
        return x, gamma, beta


class ResidualBlockTnet(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockTnet, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.spade1 = SPADEInstanceNorm2d(dim_out)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.spade2 = SPADEInstanceNorm2d(dim_out)

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        out = self.conv1(x)
        out = self.spade1(out, gamma1, beta1)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.spade2(out, gamma2, beta2)
        return x + out

    def forward_mix(self, x, param_A, param_B, w=0, mask=None):
        out = self.conv1(x)
        out = self.spade1.forward_mix(out, param_A[:2], param_B[:2], w, mask)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.spade2.forward_mix(out, param_A[2:], param_B[2:], w, mask)
        return x + out


class ResidualBlockPnet(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlockPnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            GetSPADE(dim_out, config.num_channel)
        )
        # gamma和beta都在ReLU后得到, 对conv1的输出使用s=1的卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            GetSPADE(dim_out, config.num_channel)
        )

    def forward(self, x):
        # 得到在bottleneck的两对IN参数
        out, gamma1, beta1 = self.conv1(x)
        out, gamma2, beta2 = self.conv2(out)
        assert isinstance(beta2, object)
        return x + out, gamma1, beta1, gamma2, beta2


class SPADEInstanceNorm2d(nn.Module):
    def __init__(self, conv_dim):
        super(SPADEInstanceNorm2d, self).__init__()
        self.InstanceNorm = nn.InstanceNorm2d(conv_dim, affine=False)   # 只归一化到标准正态

    def forward(self, x, gamma, beta):     # 输入w代表加入特征的浓淡
        norm = self.InstanceNorm(x)
        out = norm * (1 + gamma) + beta
        return out

    def forward_mix(self, x, param_A, param_B, w=0, mask=None):     # 输入w代表加入特征的浓淡 （A默认表示org，B表示ref）
        gamma_A = param_A[0]
        beta_A = param_A[1]
        gamma_B = param_B[0]
        beta_B = param_B[1]
        if mask is None:
            gamma = w * gamma_A + (1-w) * gamma_B
            beta = w * beta_A + (1-w) * beta_B
        else:
            mask = F.interpolate(mask, size=x.size()[2:])
            gamma = mask * gamma_A + (1-mask) * gamma_B
            beta = mask * beta_A + (1-mask) * beta_B

        norm = self.InstanceNorm(x)
        out = norm * (1 + gamma) + beta
        return out


class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False, bn_layer=True, use_diff=True):
        super(NONLocalBlock2D, self).__init__()

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        # self.softmax = nn.Softmax(dim=-1)
        self.down_sample = False
        self.use_diff = use_diff

        if self.inter_channels is None:
            self.inter_channels = in_channels//2 if in_channels>=2 else 1

        conv_nd = nn.Conv2d
        norm_layer = nn.InstanceNorm2d
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

    def forward(self, source, weight):
        """(b, c, h, w)
        src_diff: (3, 136, 32, 32)
        return: 从source中采样，得到target的形状
        """
        if self.down_sample:
            source = self.down_layer(source)

        batch_size = source.size(0)

        g_source = source.view(batch_size, self.inter_channels, -1)  # (N, C, H*W)
        g_source = g_source.permute(0, 2, 1)  # (N, H*W, C)

        y = torch.bmm(weight, g_source)
        y = y.permute(0, 2, 1).contiguous()  # (N, C, H*W)
        y = y.view(batch_size, self.inter_channels, *source.size()[2:])     # 为啥有*

        if self.down_sample:
            # (torch.where(W_y != 0, torch.zeros(W_y.shape).cuda(), W_y) * target).max() 检测为0！即刚好生成了与target相同的形状
            y = self.up_layer(y)
        return y


class Generator_spade(nn.Module):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, repeat_num=6, stage=(True, True, True), use_atten=True, use_diff=True):
        super(Generator_spade, self).__init__()
        self.repeat_num = repeat_num
        self.stage = stage
        self.use_atten = use_atten
        self.use_diff = use_diff
        self.num_conv = 2
        self.p_max = max(config.bottlenect_stage)

        # ---------------------------------------------- PNet -------------------------------------------------

        layers = nn.Sequential(
            nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True)
        )
        if self.stage[0]:
            layers = nn.Sequential(layers, GetSPADE(conv_dim, config.num_channel))
        self.pnet_in = layers

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(self.num_conv):
            layers = nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(curr_dim * 2, affine=True),
                nn.ReLU(inplace=True),
            )
            if self.stage[0]:
                layers = nn.Sequential(layers, GetSPADE(curr_dim * 2, config.num_channel))

            setattr(self, f'pnet_down_{i+1}', layers)
            curr_dim = curr_dim * 2

        # Bottleneck. All bottlenecks share the same attention module
        if self.use_atten and self.stage[1]:
            self.atten_bottleneck_g = NONLocalBlock2D(config.num_channel, use_diff=True)
            self.atten_bottleneck_b = NONLocalBlock2D(config.num_channel, use_diff=True)
            # Final Version: simple!
            self.simple_spade = GetSPADE(curr_dim, config.num_channel)

        for i in range(self.p_max+1):
            setattr(self, f'pnet_bottleneck_{i+1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='p'))

        if self.stage[2]:
            # Up-Sampling
            for i in range(self.num_conv):
                layers = nn.Sequential(
                    nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm2d(curr_dim // 2, affine=True),
                    nn.ReLU(inplace=True)
                )
                if self.stage[2]:
                    layers = nn.Sequential(layers, GetSPADE(curr_dim // 2, config.num_channel))

                setattr(self, f'pnet_up_{i+1}', nn.Sequential(*layers))
                curr_dim = curr_dim // 2

            layers = nn.Sequential(
                nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False),
                nn.Tanh()
            )
            self.pnet_out = layers

        # ------------------------------------------------- TNet -------------------------------------------------

        self.tnet_in_conv = nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3, bias=False)
        self.tnet_in_spade = SPADEInstanceNorm2d(conv_dim) if self.stage[0] else nn.InstanceNorm2d(conv_dim, affine=not config.tnet_whiten)
        self.tnet_in_relu = nn.ReLU(inplace=True)

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(self.num_conv):
            setattr(self, f'tnet_down_conv_{i+1}', nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            if self.stage[0]:
                setattr(self, f'tnet_down_spade_{i+1}', SPADEInstanceNorm2d(curr_dim*2))
            else:
                setattr(self, f'tnet_down_spade_{i+1}', nn.InstanceNorm2d(curr_dim * 2, affine=not config.tnet_whiten))

            setattr(self, f'tnet_down_relu_{i+1}', nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            setattr(self, f'tnet_bottleneck_{i+1}', ResidualBlock(dim_in=curr_dim, dim_out=curr_dim, net_mode='t'))

        # Up-Sampling
        for i in range(self.num_conv):
            setattr(self, f'tnet_up_conv_{i+1}', nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
            if self.stage[2]:
                setattr(self, f'tnet_up_spade_{i+1}', SPADEInstanceNorm2d(curr_dim//2))
            else:
                setattr(self, f'tnet_up_spade_{i+1}', nn.InstanceNorm2d(curr_dim // 2, affine=not config.tnet_whiten))
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
        HW = config.diff_size * config.diff_size
        batch_size = 3
        if config.w_visual and (fea_s is not None):   # fea_s when i==3
            # get 3 part fea using mask
            channel_num = fea_s.shape[1]

            mask_c_re = F.interpolate(mask_c, size=config.diff_size).repeat(1, channel_num, 1, 1)  # (3, c, h, w)
            fea_c = fea_c.repeat(3, 1, 1, 1)                 # (3, c, h, w)
            fea_c = fea_c * mask_c_re                        # (3, c, h, w) 最后做atten的fea。3代表3个部位。

            mask_s_re = F.interpolate(mask_s, size=config.diff_size).repeat(1, channel_num, 1, 1)
            fea_s = fea_s.repeat(3, 1, 1, 1)
            fea_s = fea_s * mask_s_re

            theta_input = torch.cat((fea_c * config.w_visual, diff_c), dim=1)
            phi_input = torch.cat((fea_s * config.w_visual, diff_s), dim=1)
            # theta_input = fea_c * config.w_visual
            # phi_input = fea_s * config.w_visual

            theta_target = theta_input.view(batch_size, -1, HW)  # (N, C+136, H*W)
            theta_target = theta_target.permute(0, 2, 1)  # (N, H*W, C)

            phi_source = phi_input.view(batch_size, -1, HW)  # (N, C+136, H*W)
        else:
            theta_target = diff_c.view(batch_size, -1, HW).permute(0, 2, 1)  # (N, H*W, 136)
            phi_source = diff_s.view(batch_size, -1, HW)  # (N, 136, H*W)

        weight = torch.bmm(theta_target, phi_source)    # (3, HW, HW)  3是batch_size，其实就是部位的个数
        one = torch.tensor(1.0)
        weight_mask = torch.where(weight == torch.tensor(0.0), weight, one)  # 取出为1的mask

        weight = weight * config.w_weight  # magic numpy，用于调节采样的范围
        # weight = weight * 10             # 用于展示可视化结果
        weight = nn.Softmax(dim=-1)(weight)
        weight = weight * weight_mask  # 不能对有grad的Tensor使用 *= 这种inplace操作！
        return weight

    def forward_atten(self, c, s, mask_list_c, mask_list_s, diff_c, diff_s, gamma=None, beta=None, ret=False):
        """attention version
        c: (b, c, h, w)
        mask_list_c: lip, skin, eye. (b, 1, h, w)
        """
        img_c = c
        img_s = s
        mask_c = torch.cat(mask_list_c, 0)      # (3, 1, h, w)
        mask_s = torch.cat(mask_list_s, 0)
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
        for i in range(self.repeat_num):
            if gamma is None and i <= self.p_max:
                cur_pnet_bottleneck = getattr(self, f'pnet_bottleneck_{i+1}')
            cur_tnet_bottleneck = getattr(self, f'tnet_bottleneck_{i+1}')

            # get s_pnet from p and transform
            if i==3:
                if gamma is None:               # not in test_mix
                    s, gamma, beta = self.simple_spade(s)
                    # if (mask_s[1] != mask_c[1]).any():  # 如果不是同一张图才atten
                    weight = self.get_weight(mask_c, mask_s, c_tnet, s, diff_c, diff_s)

                    gamma, beta = self.atten_feature(mask_s, weight, gamma, beta, self.atten_bottleneck_g, self.atten_bottleneck_b)
                    if ret:
                        return [gamma, beta]
                # else:                       # in test mode
                    # gamma, beta = param_A[0]*w + param_B[0]*(1-w), param_A[1]*w + param_B[1]*(1-w)

                c_tnet = c_tnet * (1 + gamma) + beta

            if gamma is None and i <= self.p_max:
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

    def forward(self, c, s, lms_c, lms_s, use_warp=True):
        """warping version"""
        # repeat lms to (2*b, 68, 2) for warp
        lms_c = lms_c.repeat(2, 1, 1)
        lms_s = lms_s.repeat(2, 1, 1)

        s, gamma, beta = self.pnet_in(s)
        if use_warp:
            gamma, beta = self.warp_feature(gamma, beta, lms_c, lms_s)
        c = self.tnet_in_conv(c)
        c = self.tnet_in_spade(c, gamma, beta) if self.stage[0] else self.tnet_in_spade(c)
        c = self.tnet_in_relu(c)

        # down-sampling
        for i in range(2):
            cur_pnet_down = getattr(self, f'pnet_down_{i + 1}')
            cur_tnet_down_conv = getattr(self, f'tnet_down_conv_{i + 1}')
            cur_tnet_down_spade = getattr(self, f'tnet_down_spade_{i + 1}')
            cur_tnet_down_relu = getattr(self, f'tnet_down_relu_{i + 1}')
            lms_c /= 2
            lms_s /= 2
            s, gamma, beta = cur_pnet_down(s)
            if use_warp:
                gamma, beta = self.warp_feature(gamma, beta, lms_c, lms_s)
            c = cur_tnet_down_conv(c)
            c = cur_tnet_down_spade(c, gamma, beta) if self.stage[0] else cur_tnet_down_spade(c)
            c = cur_tnet_down_relu(c)

        # bottleneck
        for i in range(self.repeat_num):
            cur_pnet_bottleneck = getattr(self, f'pnet_bottleneck_{i + 1}')
            cur_tnet_bottleneck = getattr(self, f'tnet_bottleneck_{i + 1}')
            if self.stage[1]:
                s, gamma1, beta1, gamma2, beta2 = cur_pnet_bottleneck(s)
                if use_warp:
                    gamma1, beta1, gamma2, beta2 = self.warp_feature_2(gamma1, beta1, gamma2, beta2, lms_c, lms_s)
                c = cur_tnet_bottleneck(c, gamma1, beta1, gamma2, beta2)
            else:
                s = cur_pnet_bottleneck(s)
                c = cur_tnet_bottleneck(c)

        # up-sampling
        for i in range(2):
            cur_pnet_up = getattr(self, f'pnet_up_{i + 1}')
            cur_tnet_up_conv = getattr(self, f'tnet_up_conv_{i + 1}')
            cur_tnet_up_spade = getattr(self, f'tnet_up_spade_{i + 1}')
            cur_tnet_up_relu = getattr(self, f'tnet_up_relu_{i + 1}')
            lms_c *= 2
            lms_s *= 2
            s, gamma, beta = cur_pnet_up(s)
            if use_warp:
                gamma, beta = self.warp_feature(gamma, beta, lms_c, lms_s)
            c = cur_tnet_up_conv(c)
            c = cur_tnet_up_spade(c, gamma, beta) if self.stage[2] else cur_tnet_up_spade(c)
            c = cur_tnet_up_relu(c)

        c = self.tnet_out(c)
        return c

    def forward_ori(self, c, s, mask_c=None, mask_s=None):
        """original version"""
        c = self.tnet_in_conv(c)
        if self.stage[0]:
            s, gamma, beta = self.pnet_in(s)
            c = self.tnet_in_spade(c, gamma, beta)
        else:
            s = self.pnet_in(s)
            c = self.tnet_in_spade(c)
        c = self.tnet_in_relu(c)

        # down-sampling
        for i in range(self.num_conv):  # todo: 改成3
            cur_pnet_down = getattr(self, f'pnet_down_{i + 1}')
            cur_tnet_down_conv = getattr(self, f'tnet_down_conv_{i + 1}')
            cur_tnet_down_spade = getattr(self, f'tnet_down_spade_{i + 1}')
            cur_tnet_down_relu = getattr(self, f'tnet_down_relu_{i + 1}')
            c = cur_tnet_down_conv(c)
            if self.stage[0]:
                s, gamma, beta = cur_pnet_down(s)
                c = cur_tnet_down_spade(c, gamma, beta)
            else:
                s = cur_pnet_down(s)
                c = cur_tnet_down_spade(c)
            c = cur_tnet_down_relu(c)

        # bottleneck
        if mask_c is not None:
            # mask_c = F.interpolate(mask_c[0], size=s.size()[2:]).repeat(1, 256, 1, 1)
            mask_s = F.interpolate(mask_s[0], size=s.shape[2:]).repeat(1, s.shape[1], 1, 1)
        for i in range(self.repeat_num):
            cur_pnet_bottleneck = getattr(self, f'pnet_bottleneck_{i + 1}')
            cur_tnet_bottleneck = getattr(self, f'tnet_bottleneck_{i + 1}')
            if self.stage[1]:
                s, gamma1, beta1, gamma2, beta2 = cur_pnet_bottleneck(s)
                if mask_c is not None:
                    gamma1 = gamma1 * mask_s
                    beta1 = beta1 * mask_s
                    gamma2 = gamma2 * mask_s
                    beta2 = beta2 * mask_s
                c = cur_tnet_bottleneck(c, gamma1, beta1, gamma2, beta2)
            else:
                s = cur_pnet_bottleneck(s)
                c = cur_tnet_bottleneck(c)

        # up-sampling
        for i in range(self.num_conv):
            cur_pnet_up = getattr(self, f'pnet_up_{i + 1}')
            cur_tnet_up_conv = getattr(self, f'tnet_up_conv_{i + 1}')
            cur_tnet_up_spade = getattr(self, f'tnet_up_spade_{i + 1}')
            cur_tnet_up_relu = getattr(self, f'tnet_up_relu_{i + 1}')
            c = cur_tnet_up_conv(c)
            if self.stage[2]:
                s, gamma, beta = cur_pnet_up(s)
                c = cur_tnet_up_spade(c, gamma, beta)
            else:
                s = cur_pnet_up(s)
                c = cur_tnet_up_spade(c)
            c = cur_tnet_up_relu(c)

        c = self.tnet_out(c)
        return c

    def forward_mix(self, c, param_A, param_B, w=0, mask=None):
        """style image --> PNet; content image --> TNet"""
        # in
        cnt = 0
        c = self.tnet_in_conv(c)
        c = self.tnet_in_spade.forward_mix(c, param_A[cnt], param_B[cnt], w, mask)
        c = self.tnet_in_relu(c)
        cnt += 1

        # down-sampling
        for i in range(2):
            cur_tnet_down_conv = getattr(self, f'tnet_down_conv_{i+1}')
            cur_tnet_down_spade = getattr(self, f'tnet_down_spade_{i+1}')
            cur_tnet_down_relu = getattr(self, f'tnet_down_relu_{i+1}')
            c = cur_tnet_down_conv(c)
            c = cur_tnet_down_spade.forward_mix(c, param_A[cnt], param_B[cnt], w, mask)
            c = cur_tnet_down_relu(c)
            cnt += 1

        # bottleneck
        for i in range(self.repeat_num):
            cur_tnet_bottleneck = getattr(self, f'tnet_bottleneck_{i+1}')
            c = cur_tnet_bottleneck.forward_mix(c, param_A[cnt], param_B[cnt], w, mask)
            cnt += 1

        # up-sampling
        for i in range(2):
            cur_tnet_up_conv = getattr(self, f'tnet_up_conv_{i+1}')
            cur_tnet_up_spade = getattr(self, f'tnet_up_spade_{i+1}')
            cur_tnet_up_relu = getattr(self, f'tnet_up_relu_{i+1}')
            c = cur_tnet_up_conv(c)
            c = cur_tnet_up_spade.forward_mix(c, param_A[cnt], param_B[cnt], w, mask)
            c = cur_tnet_up_relu(c)
            cnt += 1

        c = self.tnet_out(c)
        return c

    def get_spade(self, s):
        """style image --> PNet; """
        # in
        ret_list = []
        s, gamma, beta = self.pnet_in(s)
        ret_list.append((gamma, beta))

        # down-sampling
        for i in range(2):
            cur_pnet_down = getattr(self, f'pnet_down_{i+1}')
            s, gamma, beta = cur_pnet_down(s)
            ret_list.append((gamma, beta))

        # bottleneck
        for i in range(self.repeat_num):
            cur_pnet_bottleneck = getattr(self, f'pnet_bottleneck_{i+1}')
            s, gamma1, beta1, gamma2, beta2 = cur_pnet_bottleneck(s)
            ret_list.append((gamma1, beta1, gamma2, beta2))

        # up-sampling
        for i in range(2):
            cur_pnet_up = getattr(self, f'pnet_up_{i+1}')
            s, gamma, beta = cur_pnet_up(s)
            ret_list.append((gamma, beta))

        return ret_list

