from torch import nn
from torch import Tensor
import torch
import torchvision.models as models
import torch.nn.functional as F
import os
import math

"""

"""

__all__ = ["MEANet"]

model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}

mob_conv1_2 = mob_conv2_2 = mob_conv3_3 = mob_conv4_3 = mob_conv5_3 = None

def conv_1_2_hook(module, input, output):
    global mob_conv1_2
    mob_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global mob_conv2_2
    mob_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global mob_conv3_3
    mob_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global mob_conv4_3
    mob_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global mob_conv5_3
    mob_conv5_3 = output
    return None


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.mbv = models.mobilenet_v2(pretrained=True).features

        self.mbv[1].register_forward_hook(conv_1_2_hook)
        self.mbv[3].register_forward_hook(conv_2_2_hook)
        self.mbv[6].register_forward_hook(conv_3_3_hook)
        self.mbv[13].register_forward_hook(conv_4_3_hook)
        self.mbv[17].register_forward_hook(conv_5_3_hook)

    def forward(self, x: Tensor) -> Tensor:
        global mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3
        self.mbv(x)

        return mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)

class MEA(nn.Module):
    def __init__(self, channel):
        super(MEA, self).__init__()
        self.atrconv1 = BasicConv2d(channel, channel, 3, padding=3, dilation=3)
        self.atrconv2 = BasicConv2d(channel, channel, 3, padding=5, dilation=5)
        self.atrconv3 = BasicConv2d(channel, channel, 3, padding=7, dilation=7)
        self.branch1 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0))
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(channel, channel, kernel_size=(5, 1), padding=(2, 0))
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channel, channel, kernel_size=(7, 1), padding=(3, 0))
        )

        self.conv_cat1 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_cat2 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv_cat3 = BasicConv2d(2 * channel, channel, 3, padding=1)
        self.conv1_1 = BasicConv2d(channel, channel, 1)

        self.ca1 = ChannelAttention(channel)
        self.ca2 = ChannelAttention(channel)
        self.sa = SpatialAttention()
        self.edg_pred = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.sal_conv = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.branch1(x)
        x_atr1 = self.atrconv1(x)
        s_mfeb1 = self.conv_cat1(torch.cat((x1, x_atr1), 1)) + x
        x2 = self.branch2(s_mfeb1)
        x_atr2 = self.atrconv2(s_mfeb1)
        s_mfeb2 = self.conv_cat2(torch.cat((x2, x_atr2), 1)) + s_mfeb1 + x
        x3 = self.branch3(s_mfeb2)
        x_atr3 = self.atrconv3(s_mfeb2)
        s_mfeb3 = self.conv_cat3(torch.cat((x3, x_atr3), 1)) + s_mfeb1 + s_mfeb2 + x
        s_mfeb = self.conv1_1(s_mfeb3)
        s_ca = self.ca1(s_mfeb) * s_mfeb
        s_e = self.ca2(s_mfeb) * s_mfeb
        e_pred = self.edg_pred(s_e)
        s_mea = self.sal_conv((self.sa(s_ca) + self.sigmoid(e_pred)) * s_ca) + s_mfeb1 + s_mfeb2 + s_mfeb3 + x

        return s_mea, e_pred


class MSG(nn.Module):
    def __init__(self, channel):
        super(MSG, self).__init__()
        self.conv1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv2 = BasicConv2d(channel, channel, 3, padding=1)
        self.S_conv = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )

    def forward(self, fl, fh, f5, f4=None, f3=None):
        fgl1 = F.interpolate(f5, size=fl.size()[2:], mode='bilinear')
        fh = self.conv1(fgl1 * fh) + fh
        fl = self.conv2(fgl1 * fl) + fl
        out = self.S_conv(torch.cat((fh, fl), 1))
        if f4 is not None:
            fgl1 = F.interpolate(f5, size=fl.size()[2:], mode='bilinear')
            fgl2 = F.interpolate(f4, size=fl.size()[2:], mode='bilinear')
            fh = self.conv1(fgl1 * fgl2 * fh) + fh
            fl = self.conv2(fgl1 * fgl2 * fl) + fl
            out = self.S_conv(torch.cat((fh, fl), 1))
        else:
            if f4 is not None:
                if f3 is not None:
                    fgl1 = F.interpolate(f5, size=fl.size()[2:], mode='bilinear')
                    fgl2 = F.interpolate(f4, size=fl.size()[2:], mode='bilinear')
                    fgl3 = F.interpolate(f3, size=fl.size()[2:], mode='bilinear')
                    fh = self.conv1(fgl1 * fgl2 * fgl3 * fh) + fh
                    fl = self.conv2(fgl1 * fgl2 * fgl3 * fl) + fl
                    out = self.S_conv(torch.cat((fh, fl), 1))

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        return self.sigmoid(x2)

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MEANet(nn.Module):
    def __init__(self, channel=32):

        super(MEANet, self).__init__()
        self.encoder = MobileNet()
        self.reduce_sal1 = Reduction(16, channel)
        self.reduce_sal2 = Reduction(24, channel)
        self.reduce_sal3 = Reduction(32, channel)
        self.reduce_sal4 = Reduction(96, channel)
        self.reduce_sal5 = Reduction(320, channel)

        self.mea5 = MEA(channel)
        self.mea4 = MEA(channel)
        self.mea3 = MEA(channel)
        self.mea2 = MEA(channel)
        self.mea1 = MEA(channel)

        self.msg1 = MSG(channel)
        self.msg2 = MSG(channel)
        self.msg3 = MSG(channel)

        self.S1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.S5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.S_conv1 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 1)
        )
        self.trans_conv1 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                             padding=0, dilation=1, bias=False)
        self.trans_conv2 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv3 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.trans_conv4 = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                            padding=0, dilation=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        size = x.size()[2:]
        x_sal1, x_sal2, x_sal3, x_sal4, x_sal5 = self.encoder(x)

        x_sal1 = self.reduce_sal1(x_sal1)
        x_sal2 = self.reduce_sal2(x_sal2)
        x_sal3 = self.reduce_sal3(x_sal3)
        x_sal4 = self.reduce_sal4(x_sal4)
        x_sal5 = self.reduce_sal5(x_sal5)

        sal5, edg5 = self.mea5(x_sal5)
        sal4, edg4 = self.mea4(x_sal4)
        sal3, edg3 = self.mea3(x_sal3)
        sal2, edg2 = self.mea2(x_sal2)
        sal1, edg1 = self.mea1(x_sal1)

        sal4 = self.S_conv1(torch.cat([sal4, self.trans_conv1(sal5)], dim=1))
        sal3 = self.msg1(sal3, self.trans_conv2(sal4), sal5)
        sal2 = self.msg2(sal2, self.trans_conv3(sal3), sal5, sal4)
        sal1 = self.msg3(sal1, self.trans_conv4(sal2), sal5, sal4, sal3)

        sal_out = self.S1(sal1)
        sal2 = self.S2(sal2)
        sal3 = self.S3(sal3)
        sal4 = self.S4(sal4)
        sal5 = self.S5(sal5)

        sal_out = F.interpolate(sal_out, size=size, mode='bilinear', align_corners=True)
        edg_out = F.interpolate(edg1, size=size, mode='bilinear', align_corners=True)
        sal2 = F.interpolate(sal2, size=size, mode='bilinear', align_corners=True)
        edg2 = F.interpolate(edg2, size=size, mode='bilinear', align_corners=True)
        sal3 = F.interpolate(sal3, size=size, mode='bilinear', align_corners=True)
        edg3 = F.interpolate(edg3, size=size, mode='bilinear', align_corners=True)
        sal4 = F.interpolate(sal4, size=size, mode='bilinear', align_corners=True)
        edg4 = F.interpolate(edg4, size=size, mode='bilinear', align_corners=True)
        sal5 = F.interpolate(sal5, size=size, mode='bilinear', align_corners=True)
        edg5 = F.interpolate(edg5, size=size, mode='bilinear', align_corners=True)

        return sal_out, self.sigmoid(sal_out), edg_out, sal2, self.sigmoid(sal2), edg2, sal3, self.sigmoid(
            sal3), edg3, sal4, self.sigmoid(sal4), edg4, sal5, self.sigmoid(sal5), edg5