from torch import nn
import torch
from radionets.dl_framework.model import (
    conv,
    Lambda,
    flatten,
    fft,
    deconv,
    double_conv,
    flatten_with_channel,
    reshape,
    GeneralRelu,
)


def test():
    arch = nn.Sequential(Lambda(flatten), nn.Linear(7938, 7938), Lambda(reshape))
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
        Lambda(flatten),
        Lambda(fft),
        Lambda(flatten),
        nn.Linear(8192, 4096),
    )
    return arch


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
