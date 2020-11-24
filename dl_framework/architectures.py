from torch import nn
import torch
from dl_framework.model import (
    conv,
    Lambda,
    flatten,
    fft,
    deconv,
    double_conv,
    cut_off,
    flatten_with_channel,
    depth_conv,
    LocallyConnected2d,
    symmetry,
    shape,
    GeneralELU,
    conv_phase,
    conv_amp,
    vaild_gauss_bs,
    euler,
    reshape
)
from functools import partial
from math import pi

def test():
    arch = nn.Sequential(
	Lambda(flatten),
	nn.Linear(7938,7938),
	Lambda(reshape)
    )
    return arch

def cnn():
    """
    conv-layer: number of entry channels, number of exit channels,
                kerner size, stride, padding
    """
    arch = nn.Sequential(
        *conv(1, 4, (3, 3), 2, 1),
        *conv(4, 8, (3, 3), 2, 1),
        *conv(8, 16, (3, 3), 2, 1),
        nn.MaxPool2d((3, 3)),
        *conv(16, 32, (2, 2), 2, 1),
        *conv(32, 64, (2, 2), 2, 1),
        nn.MaxPool2d((2, 2)),
        Lambda(flatten),
        nn.Linear(64, 8192),
        Lambda(fft),
        Lambda(flatten),
        # *conv(2, 1, 1, 1, 0),
        nn.Linear(8192, 4096),
        # Lambda(flatten),
    )
    return arch

def small():
    """
    conv-layer: number of entry channels, number of exit channels,
                kerner size, stride, padding
    """
    arch = nn.Sequential(
        Lambda(flatten), Lambda(fft), Lambda(flatten), nn.Linear(8192, 4096),
    )
    return arch


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(*double_conv(1, 4))
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8))
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16))
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32))
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64))

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32))
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16))
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8))
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4))

        self.conv_last = nn.Conv2d(4, 1, 1)
        self.flatten = Lambda(flatten)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        x = self.conv_last(x)

        out = self.flatten(x)

        return out


def autoencoder():
    arch = nn.Sequential(
        *conv(1, 4, (3, 3), 2, 1),
        *conv(4, 8, (3, 3), 2, 1),
        *conv(8, 16, (3, 3), 2, 1),
        nn.MaxPool2d((3, 3)),
        *conv(16, 32, (2, 2), 2, 1),
        *conv(32, 64, (2, 2), 2, 1),
        nn.MaxPool2d((2, 2)),
        nn.ConvTranspose2d(64, 32, (3, 3), 2, 1, 1),
        nn.BatchNorm2d(32),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(32, 16, (3, 3), 2, 1, 1),
        nn.BatchNorm2d(16),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(16, 16, (3, 3), 2, 1, 1),
        nn.BatchNorm2d(16),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(16, 8, (3, 3), 2, 1, 1),
        nn.BatchNorm2d(8),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(8, 4, (3, 3), 2, 1, 1),
        nn.BatchNorm2d(4),
        GeneralRelu(leak=0.1, sub=0.4),
        nn.ConvTranspose2d(4, 1, (3, 3), 2, 1, 1),
        Lambda(flatten),
    )
    return arch


def autoencoder_two_channel():
    arch = nn.Sequential(
        *conv(2, 4, (3, 3), 2, 1),
        *conv(4, 8, (3, 3), 2, 1),
        *conv(8, 16, (3, 3), 2, 1),
        nn.MaxPool2d((3, 3)),
        *conv(16, 32, (2, 2), 2, 1),
        *conv(32, 64, (2, 2), 2, 1),
        nn.MaxPool2d((2, 2)),
        *deconv(64, 32, (3, 3), 2, 1, 0),
        *deconv(32, 16, (3, 3), 2, 1, 0),
        *deconv(16, 16, (3, 3), 2, 1, 0),
        *deconv(16, 8, (3, 3), 2, 1, 0),
        *deconv(8, 4, (3, 3), 2, 1, 0),
        # nn.ConvTranspose2d(4, 2, (3, 3), 2, 1, 1),
        *deconv(4, 2, (3, 3), 2, 1, 0),
        Lambda(flatten),
        # nn.Linear(8192, 4096)
        nn.Linear(2, 4096),
    )
    return arch

class UNet_fft(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)

        self.conv_last = nn.Conv2d(4, 2, 1)
        self.flatten = Lambda(flatten)
        self.linear1 = nn.Linear(8192, 4096)
        self.fft = Lambda(fft)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)

        x = self.flatten(x)
        x = self.fft(x)
        x = self.flatten(x)
        out = self.linear1(x)

        return out

class UNet_denoise(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)

        self.conv_last = nn.Conv2d(4, 2, 1)
        self.flatten = Lambda(flatten)
        self.linear = nn.Linear(7938, 3969)
        self.fft = Lambda(fft)
        self.cut = Lambda(cut_off)
        self.shape = Lambda(shape)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fft(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        x = self.flatten(x)
        out = self.linear(x)

        return out

class UNet_fourier(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)

        self.conv_last = nn.Conv2d(4, 2, 1)
        self.linear = nn.Linear(8192, 4096)
        self.flatten = Lambda(flatten)
        self.flatten_with_channel = Lambda(flatten_with_channel)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)

        x = self.upsample1(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample2(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample3(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample4(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        # out = self.flatten_with_channel(x)

        return x

class UNet_denoise_64(nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv_down1 = nn.Sequential(*double_conv(2, 4, (3, 3), 1, 1),)
        self.dconv_down2 = nn.Sequential(*double_conv(4, 8, (3, 3), 1, 1),)
        self.dconv_down3 = nn.Sequential(*double_conv(8, 16, (3, 3), 1, 1),)
        self.dconv_down4 = nn.Sequential(*double_conv(16, 32, (3, 3), 1, 1),)
        self.dconv_down5 = nn.Sequential(*double_conv(32, 64, (3, 3), 1, 1),)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dconv_up4 = nn.Sequential(*double_conv(32 + 64, 32, (3, 3), 1, 1),)
        self.dconv_up3 = nn.Sequential(*double_conv(16 + 32, 16, (3, 3), 1, 1),)
        self.dconv_up2 = nn.Sequential(*double_conv(8 + 16, 8, (3, 3), 1, 1),)
        self.dconv_up1 = nn.Sequential(*double_conv(4 + 8, 4, (3, 3), 1, 1),)
        self.conv_last = nn.Conv2d(4, 2, 1)
        self.flatten = Lambda(flatten)
        self.linear = nn.Linear(8192, 4096)
        self.fft = Lambda(fft)
        self.cut = Lambda(cut_off)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fft(x)
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)
        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        x = self.conv_last(x)
        x = self.flatten(x)
        out = self.linear(x)
        return out

def convs():
    arch = nn.Sequential(
        Lambda(flatten),
        Lambda(fft),
        *conv(2, 4, (3, 3), 2, 1),
        *conv(4, 8, (3, 3), 2, 1),
        *conv(8, 16, (3, 3), 2, 1),
        Lambda(flatten),
        nn.Linear(1024, 1024),
        Lambda(flatten_with_channel),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        *conv(16, 8, (3, 3), 1, 1),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        *conv(8, 4, (3, 3), 1, 1),
        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        *conv(4, 1, (3, 3), 1, 1),
        Lambda(flatten),
    )
    return arch

class conv_filter(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(*conv(2, 1, (5, 5), 1, 2))
        self.conv2 = nn.Sequential(*conv(1, 128, (5, 5), 1, 2))
        self.conv3 = nn.Sequential(*conv(128, 1, (5, 5), 1, 2))
        self.flatten = Lambda(flatten)

    def forward(self, x):
        x = self.conv1(x)
        inp = x.clone()
        x = self.conv2(x)
        x = self.conv3(x)
        out = x + inp
        out = self.flatten(out)

        return out

class depthwise_seperable_conv(nn.Module):
    def __init__(self):
        super().__init__()

        self.depth1 = nn.Sequential(
            *depth_conv(1, 64, (7, 7), stride=1, padding=6, dilation=2)
        )
        self.depth12 = nn.Sequential(
            *depth_conv(1, 64, (7, 7), stride=1, padding=3, dilation=1)
        )
        self.depth2 = nn.Sequential(
            *depth_conv(1, 64, (7, 7), stride=1, padding=6, dilation=2)
        )
        self.depth21 = nn.Sequential(
            *depth_conv(1, 64, (7, 7), stride=1, padding=3, dilation=1)
        )
        self.point1 = nn.Sequential(*conv(128, 1, (1, 1), 1, 0))
        self.point2 = nn.Sequential(*conv(128, 1, (1, 1), 1, 0))
        self.flatten = Lambda(flatten_with_channel)

    def forward(self, x):
        # inp = x.clone()
        inp_real = x[:, 0, :].view(x.shape[0], 1, x.shape[2], x.shape[3])
        inp_imag = x[:, 1, :].view(x.shape[0], 1, x.shape[2], x.shape[3])

        depth1 = self.depth1(inp_real)
        depth12 = self.depth12(inp_real)

        depth2 = self.depth2(inp_imag)
        depth21 = self.depth21(inp_imag)

        comb1 = torch.cat([depth1, depth12], dim=1)
        comb2 = torch.cat([depth2, depth21], dim=1)

        point1 = self.point1(comb1)
        point2 = self.point2(comb2)

        comb = torch.cat([point1, point2], dim=1)
        # comb = comb + inp
        out = self.flatten(comb)

        return comb

class small_fourier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_last = nn.Conv2d(4, 2, 1)
        self.flatten_with_channel = Lambda(flatten_with_channel)
        self.conv1 = nn.Sequential(*conv(2, 4, (3, 3), stride=1, padding=1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_last(x)
        out = self.flatten_with_channel(x)

        return out

class filter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                dilation=1,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(3, 3),
                stride=1,
                padding=2,
                dilation=2,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(5, 5),
                stride=1,
                padding=2,
                dilation=1,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(5, 5),
                stride=1,
                padding=4,
                dilation=2,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(7, 7),
                stride=1,
                padding=3,
                dilation=1,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(9, 9),
                stride=1,
                padding=4,
                dilation=1,
                padding_mode="zeros",
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
        )
        self.conv_last = nn.Sequential(
            LocallyConnected2d(192, 2, 63, 1, stride=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ELU(),
        )
        self.elu = nn.ELU()
        self.symmetry = Lambda(symmetry)

    def forward(self, x):
        inp = x.clone()
        comb = torch.cat(
            [
                self.conv1(x),
                self.conv2(x),
                self.conv3(x),
                self.conv4(x),
                self.conv5(x),
                self.conv6(x),
            ],
            dim=1,
        )
        x = self.conv_last(comb)
        x = x.clone()
        x[:, 0][inp[:, 0] != 0] = inp[:, 0][inp[:, 0] != 0]
        #  x[:, 0][inp[:, 0] == 0] += 1
        x0 = self.symmetry(x[:, 0]).reshape(-1, 1, 63, 63)
        x[:, 1][inp[:, 0] == 0] += (1e-5 + 1)
        x[:, 1][inp[:, 0] != 0] = 1e-8
        x = self.elu(x)
        x1 = self.symmetry(x[:, 1]).reshape(-1, 1, 63, 63)
        out = torch.cat([x0, x1], dim=1)
        return out

class filter_deep_amp(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.conv1_amp = nn.Sequential(
            *conv_amp(1, 4, (23, 23), 1, 11, 1)
        )
        self.conv2_amp = nn.Sequential(
             *conv_amp(4, 8, (21, 21), 1, 10, 1)
        )
        self.conv3_amp = nn.Sequential(
             *conv_amp(8, 12, (17, 17), 1, 8, 1)
        )
        self.conv_con1_amp = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.conv4_amp = nn.Sequential(
             *conv_amp(1, 4, (5, 5), 1, 4, 2)
        )
        self.conv4_phase = nn.Sequential(
             *conv_phase(1, 4, (5, 5), 1, 4, 2, add=1-pi)
        )
        self.conv5_amp = nn.Sequential(
             *conv_amp(4, 8, (5, 5), 1, 2, 1)
        )
        self.conv6_amp = nn.Sequential(
             *conv_amp(8, 12, (3, 3), 1, 2, 2)
        )
        self.conv6_phase = nn.Sequential(
             *conv_phase(8, 12, (3, 3), 1, 2, 2, add=1-pi)
        )
        self.conv7_amp = nn.Sequential(
             *conv_amp(12, 16, (3, 3), 1, 1, 1)
        )
        self.conv_con2_amp = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.conv8_amp = nn.Sequential(
             *conv_amp(1, 4, (3, 3), 1, 1, 1)
        )
        self.conv9_amp = nn.Sequential(
             *conv_amp(4, 8, (3, 3), 1, 1, 1)
        )
        self.conv10_amp = nn.Sequential(
             *conv_amp(8, 12, (3, 3), 1, 2, 2)
        )
        self.conv_con3_amp = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.symmetry_real = Lambda(symmetry)

    def forward(self, x):
        inp = x.clone()
        amp = x[:, 0, :].unsqueeze(1)

        # First block
        amp = self.conv1_amp(amp)

        amp = self.conv2_amp(amp)

        amp = self.conv3_amp(amp)

        amp = self.conv_con1_amp(amp)

        # Second block
        amp = self.conv4_amp(amp)

        amp = self.conv5_amp(amp)

        amp = self.conv6_amp(amp)

        amp = self.conv7_amp(amp)

        amp = self.conv_con2_amp(amp)

        # Third block
        amp = self.conv8_amp(amp)

        amp = self.conv9_amp(amp)

        amp = self.conv10_amp(amp)

        amp = self.conv_con3_amp(amp)

        inp_amp = inp[:, 0].unsqueeze(1)
        x0 = self.symmetry_real(amp).reshape(-1, 1, amp.shape[2], amp.shape[2])
        x0[inp_amp != 0] = inp_amp[inp_amp != 0]

        return x0

class filter_deep_phase(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.conv1_phase = nn.Sequential(
            *conv_phase(1, 4, (23, 23), 1, 11, 1, add=-2.1415)
        )
        self.conv2_phase = nn.Sequential(
             *conv_phase(4, 8, (21, 21), 1, 10, 1, add=-2.1415)
        )
        self.conv3_phase = nn.Sequential(
             *conv_phase(8, 12, (17, 17), 1, 8, 1, add=-2.1415)
        )
        self.conv_con1_phase = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(-2.1415),
        )

        self.conv4_phase = nn.Sequential(
             *conv_phase(1, 4, (5, 5), 1, 3, 2, add=-2.1415)
        )
        self.conv5_phase = nn.Sequential(
             *conv_phase(4, 8, (5, 5), 1, 2, 1, add=-2.1415)
        )
        self.conv6_phase = nn.Sequential(
             *conv_phase(8, 12, (3, 3), 1, 3, 2, add=-2.1415)
        )
        self.conv7_phase = nn.Sequential(
             *conv_phase(12, 16, (3, 3), 1, 1, 1, add=-2.1415)
        )
        self.conv_con2_phase = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(-2.1415),
        )

        self.conv8_phase = nn.Sequential(
             *conv_phase(1, 4, (3, 3), 1, 1, 1, add=-2.1415)
        )
        self.conv9_phase = nn.Sequential(
             *conv_phase(4, 8, (3, 3), 1, 1, 1, add=-2.1415)
        )
        self.conv10_phase = nn.Sequential(
             *conv_phase(8, 12, (3, 3), 1, 2, 2, add=-2.1415)
        )
        self.conv_con3_phase = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(-2.1415),
        )
        self.symmetry_imag = Lambda(partial(symmetry, mode='imag'))

    def forward(self, x):
        inp = x.clone()
        phase = x[:, 1, :].unsqueeze(1)

        # First block
        phase = self.conv1_phase(phase)

        phase = self.conv2_phase(phase)

        phase = self.conv3_phase(phase)

        phase = self.conv_con1_phase(phase)

        # Second block
        phase = self.conv4_phase(phase)

        phase = self.conv5_phase(phase)

        phase = self.conv6_phase(phase)

        phase = self.conv7_phase(phase)

        phase = self.conv_con2_phase(phase)

        # Third block
        phase = self.conv8_phase(phase)

        phase = self.conv9_phase(phase)

        phase = self.conv10_phase(phase)

        phase = self.conv_con3_phase(phase)

        inp_phase = inp[:, 1].unsqueeze(1)

        x1 = self.symmetry_imag(phase).reshape(-1, 1, phase.shape[2], phase.shape[2])
        x1[inp_phase != 0] = inp_phase[inp_phase != 0]
        return x1


class filter_deep(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.conv1_amp = nn.Sequential(
            *conv_amp(1, 4, (23, 23), 1, 11, 1)
        )
        self.conv1_phase = nn.Sequential(
            *conv_phase(1, 4, (23, 23), 1, 11, 1, add=1-pi)
        )
        self.conv2_amp = nn.Sequential(
             *conv_amp(4, 8, (21, 21), 1, 10, 1)
        )
        self.conv2_phase = nn.Sequential(
             *conv_phase(4, 8, (21, 21), 1, 10, 1, add=1-pi)
        )
        self.conv3_amp = nn.Sequential(
             *conv_amp(8, 12, (17, 17), 1, 8, 1)
        )
        self.conv3_phase = nn.Sequential(
             *conv_phase(8, 12, (17, 17), 1, 8, 1, add=1-pi)
        )
        self.conv_con1_amp = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con1_phase = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1-pi),
        )

        self.conv4_amp = nn.Sequential(
             *conv_amp(1, 4, (5, 5), 1, 3, 2)
        )
        self.conv4_phase = nn.Sequential(
             *conv_phase(1, 4, (5, 5), 1, 3, 2, add=1-pi)
        )
        self.conv5_amp = nn.Sequential(
             *conv_amp(4, 8, (5, 5), 1, 2, 1)
        )
        self.conv5_phase = nn.Sequential(
             *conv_phase(4, 8, (5, 5), 1, 2, 1, add=1-pi)
        )
        self.conv6_amp = nn.Sequential(
             *conv_amp(8, 12, (3, 3), 1, 3, 2)
        )
        self.conv6_phase = nn.Sequential(
             *conv_phase(8, 12, (3, 3), 1, 3, 2, add=1-pi)
        )
        self.conv7_amp = nn.Sequential(
             *conv_amp(12, 16, (3, 3), 1, 1, 1)
        )
        self.conv7_phase = nn.Sequential(
             *conv_phase(12, 16, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv_con2_amp = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con2_phase = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1-pi),
        )

        self.conv8_amp = nn.Sequential(
             *conv_amp(1, 4, (3, 3), 1, 1, 1)
        )
        self.conv8_phase = nn.Sequential(
             *conv_phase(1, 4, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv9_amp = nn.Sequential(
             *conv_amp(4, 8, (3, 3), 1, 1, 1)
        )
        self.conv9_phase = nn.Sequential(
             *conv_phase(4, 8, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv10_amp = nn.Sequential(
             *conv_amp(8, 12, (3, 3), 1, 2, 2)
        )
        self.conv10_phase = nn.Sequential(
             *conv_phase(8, 12, (3, 3), 1, 2, 2, add=1-pi)
        )
        self.conv11_amp = nn.Sequential(
             *conv_amp(12, 20, (3, 3), 1, 1, 1)
        )
        self.conv11_phase = nn.Sequential(
             *conv_phase(12, 20, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv_con3_amp = nn.Sequential(
            LocallyConnected2d(20, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con3_phase = nn.Sequential(
            LocallyConnected2d(20, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1-pi),
        )
        self.symmetry_real = Lambda(symmetry)
        self.symmetry_imag = Lambda(partial(symmetry, mode='imag'))
        self.flatten = Lambda(flatten)
        #self.fully_connected = nn.Linear(3969, 54)
        #self.fully_connected = nn.Linear(7938, 5)
        #self.fully_connected = nn.Linear(3969, 1)
        self.fully_connected = nn.Linear(7938, 6)
        self.vaild_gauss_bs = Lambda(vaild_gauss_bs)
        self.Relu = nn.ReLU()
        self.fft = Lambda(fft)
        self.euler = Lambda(euler)
        self.shape = Lambda(shape)

    def forward(self, x):
        inp = x.clone()
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        # First block
        amp = self.conv1_amp(amp)
        phase = self.conv1_phase(phase)

        amp = self.conv2_amp(amp)
        phase = self.conv2_phase(phase)

        amp = self.conv3_amp(amp)
        phase = self.conv3_phase(phase)

        amp = self.conv_con1_amp(amp)
        phase = self.conv_con1_phase(phase)

        # Second block
        amp = self.conv4_amp(amp)
        phase = self.conv4_phase(phase)

        amp = self.conv5_amp(amp)
        phase = self.conv5_phase(phase)

        amp = self.conv6_amp(amp)
        phase = self.conv6_phase(phase)

        amp = self.conv7_amp(amp)
        phase = self.conv7_phase(phase)

        amp = self.conv_con2_amp(amp)
        phase = self.conv_con2_phase(phase)

        # Third block
        amp = self.conv8_amp(amp)
        phase = self.conv8_phase(phase)

        amp = self.conv9_amp(amp)
        phase = self.conv9_phase(phase)

        amp = self.conv10_amp(amp)
        phase = self.conv10_phase(phase)

        amp = self.conv11_amp(amp)
        phase = self.conv11_phase(phase)

        amp = self.conv_con3_amp(amp)
        phase = self.conv_con3_phase(phase)

        # amp = amp + inp[:, 0].unsqueeze(1)
        inp_amp = inp[:, 0].unsqueeze(1)
        inp_phase = inp[:, 1].unsqueeze(1)
        # phase = phase + inp[:, 1].unsqueeze(1)
        x0 = self.symmetry_real(amp).reshape(-1, 1, amp.shape[2], amp.shape[2]) #amp
        x0[inp_amp != 0] = inp_amp[inp_amp != 0]
        #x0 = torch.exp(10* x0 -10) - 1e-10 

        x1 = self.symmetry_imag(phase).reshape(-1, 1, phase.shape[2], phase.shape[2]) #phase
        x1[inp_phase != 0] = inp_phase[inp_phase != 0]

        comb = torch.cat([x0, x1], dim=1)
        comb = self.flatten(comb)
        #comb = self.euler(comb)
        #comb = self.flatten(comb)
        #comb = self.fft(comb)
        #comb = self.flatten(comb)
        #comb = torch.sqrt(comb[:, 0:3969]**2 + comb[:, 3969:]**2)
        comb = self.fully_connected(comb)
        #comb = self.Relu(comb)
        #comb = self.fully_connected_2(comb)
        #comb = self.Relu(comb)
        #comb = self.fully_connected_3(comb)
        #comb = self.Relu(comb)
        #comb = self.fully_connected_4(comb) 
        #comb = self.vaild_gauss_bs(comb).reshape(-1, 3969)
        #comb = self.Relu(comb)
        return torch.abs(comb)


class filter_deep_xy(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.conv1_amp = nn.Sequential(
            *conv_amp(1, 4, (23, 23), 1, 11, 1)
        )
        self.conv1_phase = nn.Sequential(
            *conv_phase(1, 4, (23, 23), 1, 11, 1, add=1-pi)
        )
        self.conv2_amp = nn.Sequential(
             *conv_amp(4, 8, (21, 21), 1, 10, 1)
        )
        self.conv2_phase = nn.Sequential(
             *conv_phase(4, 8, (21, 21), 1, 10, 1, add=1-pi)
        )
        self.conv3_amp = nn.Sequential(
             *conv_amp(8, 12, (17, 17), 1, 8, 1)
        )
        self.conv3_phase = nn.Sequential(
             *conv_phase(8, 12, (17, 17), 1, 8, 1, add=1-pi)
        )
        self.conv_con1_amp = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con1_phase = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1-pi),
        )

        self.conv4_amp = nn.Sequential(
             *conv_amp(1, 4, (5, 5), 1, 3, 2)
        )
        self.conv4_phase = nn.Sequential(
             *conv_phase(1, 4, (5, 5), 1, 3, 2, add=1-pi)
        )
        self.conv5_amp = nn.Sequential(
             *conv_amp(4, 8, (5, 5), 1, 2, 1)
        )
        self.conv5_phase = nn.Sequential(
             *conv_phase(4, 8, (5, 5), 1, 2, 1, add=1-pi)
        )
        self.conv6_amp = nn.Sequential(
             *conv_amp(8, 12, (3, 3), 1, 3, 2)
        )
        self.conv6_phase = nn.Sequential(
             *conv_phase(8, 12, (3, 3), 1, 3, 2, add=1-pi)
        )
        self.conv7_amp = nn.Sequential(
             *conv_amp(12, 16, (3, 3), 1, 1, 1)
        )
        self.conv7_phase = nn.Sequential(
             *conv_phase(12, 16, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv_con2_amp = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con2_phase = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1-pi),
        )

        self.conv8_amp = nn.Sequential(
             *conv_amp(1, 4, (3, 3), 1, 1, 1)
        )
        self.conv8_phase = nn.Sequential(
             *conv_phase(1, 4, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv9_amp = nn.Sequential(
             *conv_amp(4, 8, (3, 3), 1, 1, 1)
        )
        self.conv9_phase = nn.Sequential(
             *conv_phase(4, 8, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv10_amp = nn.Sequential(
             *conv_amp(8, 12, (3, 3), 1, 2, 2)
        )
        self.conv10_phase = nn.Sequential(
             *conv_phase(8, 12, (3, 3), 1, 2, 2, add=1-pi)
        )
        self.conv11_amp = nn.Sequential(
             *conv_amp(12, 20, (3, 3), 1, 1, 1)
        )
        self.conv11_phase = nn.Sequential(
             *conv_phase(12, 20, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv_con3_amp = nn.Sequential(
            LocallyConnected2d(20, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con3_phase = nn.Sequential(
            LocallyConnected2d(20, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1-pi),
        )
        self.symmetry_real = Lambda(symmetry)
        self.symmetry_imag = Lambda(partial(symmetry, mode='imag'))
        self.flatten = Lambda(flatten)
        #self.fully_connected = nn.Linear(3969, 54)
        #self.fully_connected = nn.Linear(7938, 5)
        #self.fully_connected = nn.Linear(3969, 1)
        self.fully_connected = nn.Linear(7938, 2)
        self.vaild_gauss_bs = Lambda(vaild_gauss_bs)
        self.Relu = nn.ReLU()
        self.fft = Lambda(fft)
        self.euler = Lambda(euler)
        self.shape = Lambda(shape)

    def forward(self, x):
        inp = x.clone()
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        # First block
        amp = self.conv1_amp(amp)
        phase = self.conv1_phase(phase)

        amp = self.conv2_amp(amp)
        phase = self.conv2_phase(phase)

        amp = self.conv3_amp(amp)
        phase = self.conv3_phase(phase)

        amp = self.conv_con1_amp(amp)
        phase = self.conv_con1_phase(phase)

        # Second block
        amp = self.conv4_amp(amp)
        phase = self.conv4_phase(phase)

        amp = self.conv5_amp(amp)
        phase = self.conv5_phase(phase)

        amp = self.conv6_amp(amp)
        phase = self.conv6_phase(phase)

        amp = self.conv7_amp(amp)
        phase = self.conv7_phase(phase)

        amp = self.conv_con2_amp(amp)
        phase = self.conv_con2_phase(phase)

        # Third block
        amp = self.conv8_amp(amp)
        phase = self.conv8_phase(phase)

        amp = self.conv9_amp(amp)
        phase = self.conv9_phase(phase)

        amp = self.conv10_amp(amp)
        phase = self.conv10_phase(phase)

        amp = self.conv11_amp(amp)
        phase = self.conv11_phase(phase)

        amp = self.conv_con3_amp(amp)
        phase = self.conv_con3_phase(phase)

        # amp = amp + inp[:, 0].unsqueeze(1)
        inp_amp = inp[:, 0].unsqueeze(1)
        inp_phase = inp[:, 1].unsqueeze(1)
        # phase = phase + inp[:, 1].unsqueeze(1)
        x0 = self.symmetry_real(amp).reshape(-1, 1, amp.shape[2], amp.shape[2]) #amp
        x0[inp_amp != 0] = inp_amp[inp_amp != 0]
        #x0 = torch.exp(10* x0 -10) - 1e-10 

        x1 = self.symmetry_imag(phase).reshape(-1, 1, phase.shape[2], phase.shape[2]) #phase
        x1[inp_phase != 0] = inp_phase[inp_phase != 0]

        comb = torch.cat([x0, x1], dim=1)
        comb = self.flatten(comb)
        #comb = self.euler(comb)
        #comb = self.flatten(comb)
        #comb = self.fft(comb)
        #comb = self.flatten(comb)
        #comb = torch.sqrt(comb[:, 0:3969]**2 + comb[:, 3969:]**2)
        comb = self.fully_connected(comb)
        #comb = self.Relu(comb)
        #comb = self.fully_connected_2(comb)
        #comb = self.Relu(comb)
        #comb = self.fully_connected_3(comb)
        #comb = self.Relu(comb)
        #comb = self.fully_connected_4(comb) 
        #comb = self.vaild_gauss_bs(comb).reshape(-1, 3969)
        #comb = self.Relu(comb)
        return torch.abs(comb)

class filter_deep_sigma_xy_amp(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.conv1_amp = nn.Sequential(
            *conv_amp(1, 4, (23, 23), 1, 11, 1)
        )
        self.conv1_phase = nn.Sequential(
            *conv_phase(1, 4, (23, 23), 1, 11, 1, add=1-pi)
        )
        self.conv2_amp = nn.Sequential(
             *conv_amp(4, 8, (21, 21), 1, 10, 1)
        )
        self.conv2_phase = nn.Sequential(
             *conv_phase(4, 8, (21, 21), 1, 10, 1, add=1-pi)
        )
        self.conv3_amp = nn.Sequential(
             *conv_amp(8, 12, (17, 17), 1, 8, 1)
        )
        self.conv3_phase = nn.Sequential(
             *conv_phase(8, 12, (17, 17), 1, 8, 1, add=1-pi)
        )
        self.conv_con1_amp = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con1_phase = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1-pi),
        )

        self.conv4_amp = nn.Sequential(
             *conv_amp(1, 4, (5, 5), 1, 3, 2)
        )
        self.conv4_phase = nn.Sequential(
             *conv_phase(1, 4, (5, 5), 1, 3, 2, add=1-pi)
        )
        self.conv5_amp = nn.Sequential(
             *conv_amp(4, 8, (5, 5), 1, 2, 1)
        )
        self.conv5_phase = nn.Sequential(
             *conv_phase(4, 8, (5, 5), 1, 2, 1, add=1-pi)
        )
        self.conv6_amp = nn.Sequential(
             *conv_amp(8, 12, (3, 3), 1, 3, 2)
        )
        self.conv6_phase = nn.Sequential(
             *conv_phase(8, 12, (3, 3), 1, 3, 2, add=1-pi)
        )
        self.conv7_amp = nn.Sequential(
             *conv_amp(12, 16, (3, 3), 1, 1, 1)
        )
        self.conv7_phase = nn.Sequential(
             *conv_phase(12, 16, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv_con2_amp = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con2_phase = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1-pi),
        )

        self.conv8_amp = nn.Sequential(
             *conv_amp(1, 4, (3, 3), 1, 1, 1)
        )
        self.conv8_phase = nn.Sequential(
             *conv_phase(1, 4, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv9_amp = nn.Sequential(
             *conv_amp(4, 8, (3, 3), 1, 1, 1)
        )
        self.conv9_phase = nn.Sequential(
             *conv_phase(4, 8, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv10_amp = nn.Sequential(
             *conv_amp(8, 12, (3, 3), 1, 2, 2)
        )
        self.conv10_phase = nn.Sequential(
             *conv_phase(8, 12, (3, 3), 1, 2, 2, add=1-pi)
        )
        self.conv11_amp = nn.Sequential(
             *conv_amp(12, 20, (3, 3), 1, 1, 1)
        )
        self.conv11_phase = nn.Sequential(
             *conv_phase(12, 20, (3, 3), 1, 1, 1, add=1-pi)
        )
        self.conv_con3_amp = nn.Sequential(
            LocallyConnected2d(20, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con3_phase = nn.Sequential(
            LocallyConnected2d(20, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1-pi),
        )
        self.symmetry_real = Lambda(symmetry)
        self.symmetry_imag = Lambda(partial(symmetry, mode='imag'))
        self.flatten = Lambda(flatten)
        #self.fully_connected = nn.Linear(3969, 54)
        #self.fully_connected = nn.Linear(7938, 5)
        #self.fully_connected = nn.Linear(3969, 1)
        self.fully_connected = nn.Linear(7938, 3)
        self.vaild_gauss_bs = Lambda(vaild_gauss_bs)
        self.Relu = nn.ReLU()
        self.fft = Lambda(fft)
        self.euler = Lambda(euler)
        self.shape = Lambda(shape)

    def forward(self, x):
        inp = x.clone()
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        # First block
        amp = self.conv1_amp(amp)
        phase = self.conv1_phase(phase)

        amp = self.conv2_amp(amp)
        phase = self.conv2_phase(phase)

        amp = self.conv3_amp(amp)
        phase = self.conv3_phase(phase)

        amp = self.conv_con1_amp(amp)
        phase = self.conv_con1_phase(phase)

        # Second block
        amp = self.conv4_amp(amp)
        phase = self.conv4_phase(phase)

        amp = self.conv5_amp(amp)
        phase = self.conv5_phase(phase)

        amp = self.conv6_amp(amp)
        phase = self.conv6_phase(phase)

        amp = self.conv7_amp(amp)
        phase = self.conv7_phase(phase)

        amp = self.conv_con2_amp(amp)
        phase = self.conv_con2_phase(phase)

        # Third block
        amp = self.conv8_amp(amp)
        phase = self.conv8_phase(phase)

        amp = self.conv9_amp(amp)
        phase = self.conv9_phase(phase)

        amp = self.conv10_amp(amp)
        phase = self.conv10_phase(phase)

        amp = self.conv11_amp(amp)
        phase = self.conv11_phase(phase)

        amp = self.conv_con3_amp(amp)
        phase = self.conv_con3_phase(phase)

        # amp = amp + inp[:, 0].unsqueeze(1)
        inp_amp = inp[:, 0].unsqueeze(1)
        inp_phase = inp[:, 1].unsqueeze(1)
        # phase = phase + inp[:, 1].unsqueeze(1)
        x0 = self.symmetry_real(amp).reshape(-1, 1, amp.shape[2], amp.shape[2]) #amp
        x0[inp_amp != 0] = inp_amp[inp_amp != 0]
        #x0 = torch.exp(10* x0 -10) - 1e-10 

        x1 = self.symmetry_imag(phase).reshape(-1, 1, phase.shape[2], phase.shape[2]) #phase
        x1[inp_phase != 0] = inp_phase[inp_phase != 0]

        comb = torch.cat([x0, x1], dim=1)
        comb = self.flatten(comb)
        #comb = self.euler(comb)
        #comb = self.flatten(comb)
        #comb = self.fft(comb)
        #comb = self.flatten(comb)
        #comb = torch.sqrt(comb[:, 0:3969]**2 + comb[:, 3969:]**2)
        comb = self.fully_connected(comb)
        #comb = self.Relu(comb)
        #comb = self.fully_connected_2(comb)
        #comb = self.Relu(comb)
        #comb = self.fully_connected_3(comb)
        #comb = self.Relu(comb)
        #comb = self.fully_connected_4(comb) 
        #comb = self.vaild_gauss_bs(comb).reshape(-1, 3969)
        #comb = self.Relu(comb)
        return torch.abs(comb)
