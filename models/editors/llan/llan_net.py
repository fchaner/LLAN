import torch.nn as nn
from mmengine.model import BaseModule
import torch.nn.functional as F
from mmedit.registry import MODELS
from mmedit.models.base_archs import LRAG, HRConv
import torch.nn.utils.weight_norm as wn


@MODELS.register_module()
class LLANNet(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=48,
                 up_channels=24,
                 num_blocks=5,
                 num_groups=4,
                 upscale_factor=2,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.upscale_factor = upscale_factor
        self.conv_first = wn(nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=True))
        self.map_body = nn.ModuleList()
        for _ in range(num_groups):
            self.map_body.append(LRAG(mid_channels, num_blocks))
        self.trunk_conv = wn(nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True))
        self.HR_conv1 = HRConv(mid_channels, up_channels)
        if self.upscale_factor == 4:
            self.HR_conv2 = HRConv(up_channels, up_channels)
        self.conv_last = wn(nn.Conv2d(up_channels, out_channels, 3, 1, 1, bias=True))

    def forward(self, x):
        fea = self.conv_first(x)
        ind = fea
        for blo in self.map_body:
            fea = blo(fea)
        trunk = self.trunk_conv(fea)
        fea = ind + trunk
        if self.upscale_factor == 4:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
        else:
            fea = F.interpolate(fea, scale_factor=self.upscale_factor, mode='nearest')
        fea = self.HR_conv1(fea)
        if self.upscale_factor == 4:
            fea = F.interpolate(fea, scale_factor=2, mode='nearest')
            fea = self.HR_conv2(fea)
        out = self.conv_last(fea)
        ilr = F.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        out = out + ilr
        return out


