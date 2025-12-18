import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

#以此绝对路径导入官方模块，确保解耦
from ultralytics.nn.modules.block import C2f, Bottleneck
from ultralytics.nn.modules.conv import Conv, autopad

class DeformConv(nn.Module):
    """
    Deformable Convolution v2 (DCNv2) Module.
    Standard implementation using torchvision.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(DeformConv, self).__init__()
        
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.groups = groups
        
        # Offset and Mask generator
        self.offset_mask_conv = nn.Conv2d(
            in_channels, 
            3 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=True
        )
        
        # Initialize weights to 0 for stability
        nn.init.constant_(self.offset_mask_conv.weight, 0)
        nn.init.constant_(self.offset_mask_conv.bias, 0)
        
        self.dcn = DeformConv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        
    def forward(self, x):
        x = x.to(self.offset_mask_conv.weight.dtype)
        out = self.offset_mask_conv(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return self.dcn(x, offset, mask)

class Bottleneck_DCN(Bottleneck):
    """
    Standard Bottleneck but replaces the 3x3 Conv with DeformConv.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        # Override cv2 with DeformConv
        self.cv2 = nn.Sequential(
            DeformConv(c_, c2, kernel_size=k[1], stride=1, groups=g),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )

class C2f_DCN(C2f):
    """
    C2f module with Deformable Convolution in the Bottleneck.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e) 
        self.m = nn.ModuleList(Bottleneck_DCN(c_, c_, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))