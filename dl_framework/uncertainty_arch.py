from torch import nn
from dl_framework.model import GeneralELU, LocallyConnected2d
from math import pi


def conv(ni, nc, ks, stride, padding, dilation, act):
    conv = (
        nn.Conv2d(
            in_channels=ni,
            out_channels=nc,
            kernel_size=(ks, ks),
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            padding_mode="replicate",
        ),
    )
    bn = (nn.BatchNorm2d(nc),)
    layers = [*conv, *bn, act]
    return layers


def block_1_p_32():
    layers = [
        *conv(1, 4, 19, 1, 9, 1, GeneralELU(add=-(pi - 1))),
        *conv(4, 8, 17, 1, 8, 1, GeneralELU(add=-(pi - 1))),
        *conv(8, 12, 15, 1, 7, 1, GeneralELU(add=-(pi - 1))),
        LocallyConnected2d(12, 1, 32, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1)),
    ]
    return nn.Sequential(*layers)


def block_2_p_32():
    layers = [
        *conv(1, 8, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        *conv(8, 16, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        *conv(16, 24, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        LocallyConnected2d(24, 1, 32, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1)),
    ]
    return nn.Sequential(*layers)


def local_block():
    layers = [
        LocallyConnected2d(24, 1, 32, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1)),
    ]
    return nn.Sequential(*layers)


def local_p_32():
    layers = [
        LocallyConnected2d(48, 1, 32, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1), maxv=pi),
    ]
    return nn.Sequential(*layers)


def block_1_a():
    layers = [
        *conv(1, 4, 23, 1, 11, 1, GeneralELU(add=+1)),
        *conv(4, 8, 21, 1, 10, 1, GeneralELU(add=+1)),
        *conv(8, 12, 17, 1, 8, 1, GeneralELU(add=+1)),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1),
    ]
    return nn.Sequential(*layers)


def block_2_a():
    layers = [
        *conv(1, 4, 5, 1, 2, 1, GeneralELU(add=+1)),
        *conv(4, 8, 5, 1, 4, 2, GeneralELU(add=+1)),
        *conv(8, 12, 3, 1, 1, 1, GeneralELU(add=+1)),
        *conv(12, 16, 3, 1, 2, 2, GeneralELU(add=+1)),
        LocallyConnected2d(16, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1),
    ]
    return nn.Sequential(*layers)


def block_3_a():
    layers = [
        *conv(1, 4, 3, 1, 1, 1, GeneralELU(add=+1)),
        *conv(4, 8, 3, 1, 1, 1, GeneralELU(add=+1)),
        *conv(8, 12, 3, 1, 2, 2, GeneralELU(add=+1)),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1),
    ]
    return nn.Sequential(*layers)


def block_1_a_unc():
    layers = [
        *conv(1, 4, 23, 1, 11, 1, GeneralELU(add=+1+1e-5)),
        *conv(4, 8, 21, 1, 10, 1, GeneralELU(add=+1+1e-5)),
        *conv(8, 12, 17, 1, 8, 1, GeneralELU(add=+1+1e-5)),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1+1e-5),
    ]
    return nn.Sequential(*layers)


def block_2_a_unc():
    layers = [
        *conv(1, 4, 5, 1, 2, 1, GeneralELU(add=+1+1e-5)),
        *conv(4, 8, 5, 1, 4, 2, GeneralELU(add=+1+1e-5)),
        *conv(8, 12, 3, 1, 1, 1, GeneralELU(add=+1+1e-5)),
        *conv(12, 16, 3, 1, 2, 2, GeneralELU(add=+1+1e-5)),
        LocallyConnected2d(16, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1+1e-5),
    ]
    return nn.Sequential(*layers)


def block_3_a_unc():
    layers = [
        *conv(1, 4, 3, 1, 1, 1, GeneralELU(add=+1+1e-5)),
        *conv(4, 8, 3, 1, 1, 1, GeneralELU(add=+1+1e-5)),
        *conv(8, 12, 3, 1, 2, 2, GeneralELU(add=+1+1e-5)),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1+1e-5),
    ]
    return nn.Sequential(*layers)


def block_1_p():
    layers = [
        *conv(1, 4, 23, 1, 11, 1, GeneralELU(add=-(pi - 1))),
        *conv(4, 8, 21, 1, 10, 1, GeneralELU(add=-(pi - 1))),
        *conv(8, 12, 17, 1, 8, 1, GeneralELU(add=-(pi - 1))),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1)),
    ]
    return nn.Sequential(*layers)


def block_2_p():
    layers = [
        *conv(1, 8, 5, 1, 2, 1, GeneralELU(add=-(pi - 1))),
        *conv(8, 16, 5, 1, 4, 2, GeneralELU(add=-(pi - 1))),
        *conv(16, 24, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        *conv(24, 32, 3, 1, 2, 2, GeneralELU(add=-(pi - 1))),
        LocallyConnected2d(32, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1)),
    ]
    return nn.Sequential(*layers)


def block_3_p():
    layers = [
        *conv(1, 8, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        *conv(8, 16, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        *conv(16, 24, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        LocallyConnected2d(24, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1)),
    ]
    return nn.Sequential(*layers)


def bridge():
    layers = [
        LocallyConnected2d(2, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1)),
    ]
    return nn.Sequential(*layers)


def block_4_p():
    layers = [
        *conv(1, 8, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        *conv(8, 16, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        *conv(16, 24, 3, 1, 2, 2, GeneralELU(add=-(pi - 1))),
        LocallyConnected2d(24, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1)),
    ]
    return nn.Sequential(*layers)


def block_1_p_unc():
    layers = [
        *conv(1, 4, 23, 1, 11, 1, GeneralELU(add=+(1 + 1e-5))),
        *conv(4, 8, 21, 1, 10, 1, GeneralELU(add=+(1 + 1e-5))),
        *conv(8, 12, 17, 1, 8, 1, GeneralELU(add=+(1 + 1e-5))),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1 + 1e-5),
    ]
    return nn.Sequential(*layers)


def block_2_p_unc():
    layers = [
        *conv(1, 4, 5, 1, 2, 1, GeneralELU(add=+(1 + 1e-5))),
        *conv(4, 8, 5, 1, 4, 2, GeneralELU(add=+(1 + 1e-5))),
        *conv(8, 12, 3, 1, 1, 1, GeneralELU(add=+(1 + 1e-5))),
        *conv(12, 16, 3, 1, 2, 2, GeneralELU(add=+(1 + 1e-5))),
        LocallyConnected2d(16, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1 + 1e-5),
    ]
    return nn.Sequential(*layers)


def block_3_p_unc():
    layers = [
        *conv(1, 4, 3, 1, 1, 1, GeneralELU(add=+(1 + 1e-5))),
        *conv(4, 8, 3, 1, 1, 1, GeneralELU(add=+(1 + 1e-5))),
        *conv(8, 12, 3, 1, 2, 2, GeneralELU(add=+(1 + 1e-5))),
        LocallyConnected2d(12, 1, 63, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=+1 + 1e-5),
    ]
    return nn.Sequential(*layers)


def block_1_p_31():
    layers = [
        *conv(1, 4, 19, 1, 9, 1, GeneralELU(add=-(pi - 1))),
        *conv(4, 8, 17, 1, 8, 1, GeneralELU(add=-(pi - 1))),
        *conv(8, 12, 15, 1, 7, 1, GeneralELU(add=-(pi - 1))),
        LocallyConnected2d(12, 1, 31, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1)),
    ]
    return nn.Sequential(*layers)


def block_2_p_31():
    layers = [
        *conv(1, 8, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        *conv(8, 16, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        *conv(16, 24, 3, 1, 1, 1, GeneralELU(add=-(pi - 1))),
        LocallyConnected2d(24, 1, 31, 1, stride=1, bias=False),
        nn.BatchNorm2d(1),
        GeneralELU(add=-(pi - 1)),
    ]
    return nn.Sequential(*layers)
