from torch import nn
import torch
from radionets.dl_framework.model import (
    conv,
    Lambda,
    flatten,
    fft,
    double_conv,
    cut_off,
    flatten_with_channel,
    shape,
    unsqueeze1,
    squeeze1,
)


class UNet_segamp(nn.Module):
    def __init__(self):
        super().__init__()

        self.unsqueeze = Lambda(unsqueeze1)
        self.squeeze = Lambda(squeeze1)

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

        self.dconv_cnn1 = nn.Sequential(
            *conv(2, 256),
            self.maxpool,
            *conv(256, 256),
            self.maxpool,
        )
        self.dconv_cnn2 = nn.Sequential(
            *conv(256, 512),
            self.maxpool,
            *conv(512, 512),
            self.maxpool,
        )
        self.conv_cnn = nn.Sequential(*conv(512, 1024))

        self.flatten = Lambda(flatten)

        self.lin_cnn = nn.Sequential(
            nn.Linear(1024*4**2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.lin_last_cnn = nn.Sequential(nn.Linear(256, 1))

    def forward(self, x):
        inp_unet = self.unsqueeze(x)

        conv1 = self.dconv_down1(inp_unet)
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
        out_unet = torch.sigmoid(x)
        out_unet = self.squeeze(out_unet)

        inp_unet = self.squeeze(inp_unet)
        inp_cnn = torch.stack([inp_unet, out_unet], dim=1)

        x = self.dconv_cnn1(inp_cnn)
        x = self.dconv_cnn2(x)

        x = self.conv_cnn(x)

        x = self.flatten(x)

        x = self.lin_cnn(x)
        out_cnn = self.lin_last_cnn(x)

        return self.flatten(out_unet), out_cnn


class UNet_seg(nn.Module):
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
        self.unsqueeze = Lambda(unsqueeze1)

    def forward(self, x):
        conv1 = self.dconv_down1(self.unsqueeze(x))
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

        return torch.sigmoid(out)


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


class UNet_fft(nn.Module):
    def __init__(self):
        super().__init__()

        self.dconv_down1 = nn.Sequential(
            *double_conv(2, 4, (3, 3), 1, 1),
        )
        self.dconv_down2 = nn.Sequential(
            *double_conv(4, 8, (3, 3), 1, 1),
        )
        self.dconv_down3 = nn.Sequential(
            *double_conv(8, 16, (3, 3), 1, 1),
        )
        self.dconv_down4 = nn.Sequential(
            *double_conv(16, 32, (3, 3), 1, 1),
        )
        self.dconv_down5 = nn.Sequential(
            *double_conv(32, 64, (3, 3), 1, 1),
        )

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(
            *double_conv(32 + 64, 32, (3, 3), 1, 1),
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(16 + 32, 16, (3, 3), 1, 1),
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(8 + 16, 8, (3, 3), 1, 1),
        )
        self.dconv_up1 = nn.Sequential(
            *double_conv(4 + 8, 4, (3, 3), 1, 1),
        )

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

        self.dconv_down1 = nn.Sequential(
            *double_conv(2, 4, (3, 3), 1, 1),
        )
        self.dconv_down2 = nn.Sequential(
            *double_conv(4, 8, (3, 3), 1, 1),
        )
        self.dconv_down3 = nn.Sequential(
            *double_conv(8, 16, (3, 3), 1, 1),
        )
        self.dconv_down4 = nn.Sequential(
            *double_conv(16, 32, (3, 3), 1, 1),
        )
        self.dconv_down5 = nn.Sequential(
            *double_conv(32, 64, (3, 3), 1, 1),
        )

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(
            *double_conv(32 + 64, 32, (3, 3), 1, 1),
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(16 + 32, 16, (3, 3), 1, 1),
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(8 + 16, 8, (3, 3), 1, 1),
        )
        self.dconv_up1 = nn.Sequential(
            *double_conv(4 + 8, 4, (3, 3), 1, 1),
        )

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

        self.dconv_down1 = nn.Sequential(
            *double_conv(2, 4, (3, 3), 1, 1),
        )
        self.dconv_down2 = nn.Sequential(
            *double_conv(4, 8, (3, 3), 1, 1),
        )
        self.dconv_down3 = nn.Sequential(
            *double_conv(8, 16, (3, 3), 1, 1),
        )
        self.dconv_down4 = nn.Sequential(
            *double_conv(16, 32, (3, 3), 1, 1),
        )
        self.dconv_down5 = nn.Sequential(
            *double_conv(32, 64, (3, 3), 1, 1),
        )

        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(size=7, mode="bilinear", align_corners=True)
        self.upsample2 = nn.Upsample(size=15, mode="bilinear", align_corners=True)
        self.upsample3 = nn.Upsample(size=31, mode="bilinear", align_corners=True)
        self.upsample4 = nn.Upsample(size=63, mode="bilinear", align_corners=True)

        self.dconv_up4 = nn.Sequential(
            *double_conv(32 + 64, 32, (3, 3), 1, 1),
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(16 + 32, 16, (3, 3), 1, 1),
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(8 + 16, 8, (3, 3), 1, 1),
        )
        self.dconv_up1 = nn.Sequential(
            *double_conv(4 + 8, 4, (3, 3), 1, 1),
        )

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
        self.dconv_down1 = nn.Sequential(
            *double_conv(2, 4, (3, 3), 1, 1),
        )
        self.dconv_down2 = nn.Sequential(
            *double_conv(4, 8, (3, 3), 1, 1),
        )
        self.dconv_down3 = nn.Sequential(
            *double_conv(8, 16, (3, 3), 1, 1),
        )
        self.dconv_down4 = nn.Sequential(
            *double_conv(16, 32, (3, 3), 1, 1),
        )
        self.dconv_down5 = nn.Sequential(
            *double_conv(32, 64, (3, 3), 1, 1),
        )
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dconv_up4 = nn.Sequential(
            *double_conv(32 + 64, 32, (3, 3), 1, 1),
        )
        self.dconv_up3 = nn.Sequential(
            *double_conv(16 + 32, 16, (3, 3), 1, 1),
        )
        self.dconv_up2 = nn.Sequential(
            *double_conv(8 + 16, 8, (3, 3), 1, 1),
        )
        self.dconv_up1 = nn.Sequential(
            *double_conv(4 + 8, 4, (3, 3), 1, 1),
        )
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
