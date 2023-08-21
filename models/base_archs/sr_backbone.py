# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from torch import Tensor
import torch
from ..utils import default_init_weights
from mmedit.models.utils import make_layer
import torch.nn.utils.weight_norm as wn
# def default_init_weights(module, scale=1):
#     """Initialize network weights.

#     Args:
#         modules (nn.Module): Modules to be initialized.
#         scale (float): Scale initialized weights, especially for residual
#             blocks. Default: 1.
#     """
#     for m in module.modules():
#         if isinstance(m, nn.Conv2d):
#             kaiming_init(m, a=0, mode='fan_in', bias=0)
#             m.weight.data *= scale
#         elif isinstance(m, nn.Linear):
#             kaiming_init(m, a=0, mode='fan_in', bias=0)
#             m.weight.data *= scale
#         elif isinstance(m, _BatchNorm):
#             constant_init(m.weight, val=1, bias=0)

# def make_layer(block, num_blocks, **kwarg):
#     """Make layers by stacking the same blocks.

#     Args:
#         block (nn.module): nn.module class for basic block.
#         num_blocks (int): number of blocks.

#     Returns:
#         nn.Sequential: Stacked blocks in nn.Sequential.
#     """
#     layers = []
#     for _ in range(num_blocks):
#         layers.append(block(**kwarg))
#     return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:

    ::

        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Used to scale the residual before addition.
            Default: 1.0.
    """

    def __init__(self, mid_channels: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # if res_scale < 1.0, use the default initialization, as in EDSR.
        # if res_scale = 1.0, use scaled kaiming_init, as in MSRResNet.
        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for ResidualBlockNoBN.

        Initialization methods like `kaiming_init` are for VGG-style modules.
        For modules with residual paths, using smaller std is better for
        stability and performance. We empirically use 0.1. See more details in
        "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks"
        """

        for m in [self.conv1, self.conv2]:
            default_init_weights(m, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class PixA(nn.Module):
    def __init__(self, mid_channels):
        super().__init__()
        self.conv1 = wn(nn.Conv2d(mid_channels, mid_channels, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv1(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out


class LSCPA(nn.Module):
    def __init__(self, mid_channels, way_num=2):
        super().__init__()
        for_channels = int(mid_channels / way_num)
        pos_channels = int(for_channels / way_num)
        self.conv0_a = wn(nn.Conv2d(mid_channels, pos_channels, kernel_size=1))
        self.conv0_b = wn(nn.Conv2d(mid_channels, pos_channels, kernel_size=1))
        self.conv0_c = wn(nn.Conv2d(mid_channels, for_channels, kernel_size=1))
        self.conv1_a = wn(nn.Conv2d(pos_channels, pos_channels, 3, 1, 1, bias=True))
        self.conv1_b = PixA(pos_channels)
        self.conv1_c = wn(nn.Conv2d(for_channels, for_channels, 3, 1, 1, bias=True))
        self.conv2_c = wn(nn.Conv2d(for_channels, for_channels, 3, 1, 1, bias=True))
        self.conv3 = wn(nn.Conv2d(mid_channels, mid_channels, kernel_size=1))
        self.LRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out0 = self.LRelu(self.conv0_a(x))
        out1 = self.LRelu(self.conv0_b(x))
        out2 = self.LRelu(self.conv0_c(x))
        out0 = self.conv1_a(out0)
        out1 = self.conv1_b(out1)
        out2 = self.conv2_c(self.LRelu(self.conv1_c(out2)))
        out = self.conv3(torch.cat((out0, out1, out2), 1))
        return out + x


def glo_avg_pool2d(inputs):
    pool2d = torch.mean(inputs.data, dim=[2, 3], keepdim=True)
    return pool2d


class ALocA(nn.Module):
    def __init__(self, mid_channels, reduction=16):
        super().__init__()
        self.loc_a = nn.Sequential(
            wn(nn.Conv2d(mid_channels, mid_channels // reduction, 1, padding=0, bias=True)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            wn(nn.Conv2d(mid_channels // reduction, mid_channels, 1, padding=0, bias=True)),
            nn.Sigmoid()
        )

    def forward(self, x):
        hca_mean = glo_avg_pool2d(x)
        avg_loc_a = self.loc_a(hca_mean + x)
        out = x * avg_loc_a
        return out


class LRAG(nn.Module):
    def __init__(self, mid_channels: int = 48, blo_numbers=5):
        super().__init__()
        self.map_body = make_layer(
            LSCPA,
            blo_numbers,
            mid_channels=mid_channels)
        self.ALAB = ALocA(mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.map_body(x)
        out = self.ALAB(out)
        out = out + identity
        return out


class HRConv(nn.Module):
    def __init__(self, mid_channels, up_channels):
        super().__init__()
        self.conv0 = wn(nn.Conv2d(mid_channels, up_channels, 3, 1, 1, bias=True))
        self.conv1 = wn(nn.Conv2d(up_channels, up_channels, 3, 1, 1, bias=True))
        self.LRelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.LRelu(self.conv0(x))
        out = self.LRelu(self.conv1(out))
        return out

