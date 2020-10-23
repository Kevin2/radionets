from torch import nn
import torch
from dl_framework.model import (
    conv,
    Lambda,
    flatten,
    flatten_with_channel,
    depth_conv,
    LocallyConnected2d,
    symmetry,
)


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
        x[:, 1][inp[:, 0] == 0] += 1e-5 + 1
        x[:, 1][inp[:, 0] != 0] = 1e-8
        x = self.elu(x)
        x1 = self.symmetry(x[:, 1]).reshape(-1, 1, 63, 63)
        out = torch.cat([x0, x1], dim=1)
        return out
