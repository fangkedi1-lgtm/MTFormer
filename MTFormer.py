import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class Conv3BN(nn.Module):
    def __init__(self, in_: int, out: int, bn=False):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.bn = nn.BatchNorm2d(out) if bn else None
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x


class NetModule(nn.Module):
    def __init__(self, in_: int, out: int):
        super().__init__()
        self.l1 = Conv3BN(in_, out)
        self.l2 = Conv3BN(out, out)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups=64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = Parameter(torch.ones(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cSE(nn.Module):
    def __init__(self, in_channels: int, r: int = 16):
        super().__init__()
        self.linear1 = nn.Linear(in_channels, in_channels // r)
        self.linear2 = nn.Linear(in_channels // r, in_channels)

    def forward(self, x: torch.Tensor):
        input_x = x
        x = x.view(*(x.shape[:-2]), -1).mean(-1)
        x = F.relu(self.linear1(x), inplace=True)
        x = self.linear2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = torch.sigmoid(x)
        x = torch.mul(input_x, x)
        return x


class sSE(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor):
        input_x = x
        x = self.conv(x)
        x = torch.sigmoid(x)
        x = torch.mul(input_x, x)
        return x


class scSE(nn.Module):
    def __init__(self, in_channels: int, r: int = 16):
        super().__init__()
        self.cse_block = cSE(in_channels, r)
        self.sse_block = sSE(in_channels)

    def forward(self, x: torch.Tensor):
        cse = self.cse_block(x)
        sse = self.sse_block(x)
        x = torch.add(cse, sse)
        return x


class RPBlock(nn.Module):
    """Residual Projection Block"""
    def __init__(self, in_channels, out_channels):
        super(RPBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        w = torch.sigmoid(self.proj(self.pool(out)))
        out = out * w + out
        return self.relu(out + residual)


class MTBlock(nn.Module):
    """Multi-Scale Transformer Block"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super(MTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(inplace=True),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(2, 0, 1)
        x2 = self.norm1(x_flat)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x_flat + attn_out
        x2 = self.norm2(x)
        x2 = self.mlp(x2)
        x = x + x2
        x = x.permute(1, 2, 0).view(B, C, H, W)
        return x


class RSBlock(nn.Module):
    """Residual Spatial Block"""
    def __init__(self, in_channels, out_channels):
        super(RSBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + x)


class JointDeepSupervision(nn.Module):
    """Deep Supervision Head for Multi-scale Features"""
    def __init__(self, in_channels_list, num_classes=1):
        super(JointDeepSupervision, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(ch, num_classes, kernel_size=1) for ch in in_channels_list
        ])

    def forward(self, features):
        out = 0
        target_size = features[-1].shape[2:]
        for f, conv in zip(features, self.convs):
            out = out + F.interpolate(
                conv(f), size=target_size, mode='bilinear', align_corners=False)
        return out / len(features)


class MTFormer(nn.Module):
    """
    MTFormer: Joint Uncertainty-Aware Abrupt-Change Distance Transform 
    and Multi-task Transformer for Agricultural Parcel Segmentation.
    """
    def __init__(self, in_channels=3, base_channels=64, num_stages=5, num_classes=1):
        super(MTFormer, self).__init__()
        
        self.enc_blocks = nn.ModuleList()
        for i in range(num_stages):
            ch_in = in_channels if i == 0 else base_channels * (2 ** (i - 1))
            ch_out = base_channels * (2 ** i)
            self.enc_blocks.append(RPBlock(ch_in, ch_out))
            
        self.mt_blocks = nn.ModuleList([
            MTBlock(base_channels * (2 ** i)) for i in range(num_stages)
        ])
        
        self.up_convs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_channels_list = []
        
        for i in reversed(range(num_stages)):
            ch_skip = base_channels * (2 ** i)
            ch_in = base_channels * (2 ** i) if i == num_stages - 1 else base_channels * (2 ** (i + 1))
            
            self.up_convs.append(nn.Conv2d(ch_in, ch_skip, kernel_size=1))
            self.dec_blocks.append(RSBlock(ch_skip, ch_skip))
            dec_channels_list.append(ch_skip)

        self.supervision = JointDeepSupervision(dec_channels_list, num_classes)

    def forward(self, x):
        enc_feats = []
        out = x
        
        for rp, mt in zip(self.enc_blocks, self.mt_blocks):
            out = rp(out)
            out = mt(out)
            enc_feats.append(out)
            out = F.max_pool2d(out, 2)
            
        dec_feats = []
        for i, (up, rs) in enumerate(zip(self.up_convs, self.dec_blocks)):
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
            out = up(out)
            skip = enc_feats[-(i + 1)]
            out = out + skip
            out = rs(out)
            dec_feats.append(out)
            
        pred = self.supervision(dec_feats)
        return pred


if __name__ == "__main__":
    # Test network construction and forward pass
    model = MTFormer(in_channels=3, num_classes=1)
    
    # Input tensor shape: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(2, 3, 256, 256) 
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
