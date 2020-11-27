from torch import nn
from torchvision.transforms import CenterCrop
from radionets.dl_framework.model import (
    conv,
    double_conv,
    Lambda,
    flatten,
    unsqueeze1,
    absolute,
    normalization,
    clamp,
    retrieve,
)


def list_pos_():
    """
    Conv-Network with source list as target. Position only. FFT beforehand.
    """
    arch = nn.Sequential(
        Lambda(normalization),
        Lambda(unsqueeze1),
        *conv(1, 3, (3, 3), 1, 1),  # 5*63*63
        *conv(3, 6, (3, 3), 1, 1),  # 25*63*63
        *conv(6, 12, (3, 3), 1, 1),  # 12 *63*63
        Lambda(flatten),
        nn.Linear(47628, 4700),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4700, 470),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(470, 7 * 2),
        Lambda(clamp),
        # Lambda(round_),
    )
    return arch


def list_pos():
    """
    Conv-Network with source list as target. Position only. FFT beforehand.
    Best attempt at LIST position problem.
    """
    arch = nn.Sequential(
        Lambda(normalization),
        Lambda(unsqueeze1),
        *conv(1, 3, (3, 3), 1, 1),  # 3*63*63
        *conv(3, 6, (3, 3), 1, 1),  # 6*63*63
        Lambda(flatten),
        nn.Linear(23814, 2300),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(2300, 230),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(230, 5 * 2),
        Lambda(clamp),
    )
    return arch


def small():
    arch = nn.Sequential(
        CenterCrop(5),
        Lambda(flatten),
        nn.Linear(25, 225),
        nn.ReLU(),
        nn.Linear(225, 2500),
        nn.ReLU(),
        nn.Linear(2500, 5000),
        nn.ReLU(),
        nn.Linear(5000, 2500),
        nn.ReLU(),
        nn.Linear(2500, 225),
        nn.ReLU(),
        nn.Linear(225, 25),
        nn.ReLU(),
        nn.Linear(25 , 1),
        Lambda(absolute),
    )
    return arch


class Cnn_amp(nn.Module):
    def __init__(self):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_1 = nn.Sequential(
            *conv(2, 256),
            self.maxpool,
            *conv(256, 256),
            self.maxpool,
        )
        self.dconv_2 = nn.Sequential(
            *conv(256, 512),
            self.maxpool,
            *conv(512, 512),
            self.maxpool,
        )
        self.conv = nn.Sequential(*conv(512, 1024))
        self.flatten = Lambda(flatten)
        self.lin = nn.Sequential(
            nn.Linear(1024*4**2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.last_lin = nn.Sequential(nn.Linear(256, 1))
        self.absolute = Lambda(absolute)

    def forward(self, x):
        x = self.dconv_1(x)
        x = self.dconv_2(x)
        x = self.conv(x)

        x = self.flatten(x)

        x = self.lin(x)
        out = self.last_lin(x)

        return self.absolute(out)


def cnn_amp():
    """
    Best attempt at amplitude problem. Tested on one source imgs.
    """
    arch = nn.Sequential(
        Lambda(unsqueeze1),
        *conv(1, 256),
        nn.MaxPool2d(2),
        *conv(256, 256),
        nn.MaxPool2d(2),
        *conv(256, 512),
        nn.MaxPool2d(2),
        *conv(512, 512),
        nn.MaxPool2d(2),
        *conv(512, 1024),
        Lambda(flatten),
        nn.Linear(1024*3**2, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        Lambda(absolute),
    )
    return arch


def amp():
    """
    Predict amplitudes for one source images.
    """
    arch = nn.Sequential(
        Lambda(unsqueeze1),
        nn.MaxPool2d(2),#31*31
        *conv(1, 5),
        nn.MaxPool2d(2), #5*15*15
        *conv(5, 10),
        nn.MaxPool2d(2),#7*7*10
        *conv(10, 20),
        Lambda(flatten),
        nn.Linear(980, 500),
        nn.ReLU(),
        nn.Linear(500, 240),
        nn.ReLU(),
        nn.Linear(240, 120),
        nn.ReLU(),
        nn.Linear(120, 25),
        nn.ReLU(),
        nn.Linear(25, 1),
        Lambda(absolute),
    )
    return arch


def cnn_list():
    """
    conv-layers with source list as target, do fft beforehand so input=dirtyim.
    """
    arch = nn.Sequential(
        Lambda(unsqueeze1),
        *conv(1, 5, (3, 3), 1, 0),
        *conv(5, 25, (3, 3), 1, 0),
        nn.Dropout2d(),
        nn.MaxPool2d((3, 3), 2),  # 25*29*29
        Lambda(flatten),
        nn.Linear(21025, 5000),
        nn.ReLU(),
        nn.Linear(5000, 500),
        nn.ReLU(),
        nn.Linear(500, 5 * 5),
        Lambda(absolute),
    )
    return arch


def cnn_list_big():
    """
    Big arch; Conv-layers with source list as target. FFT beforehand.
    """
    arch = nn.Sequential(
        Lambda(unsqueeze1),
        *conv(1, 5, (3, 3), 1, 0),
        *conv(5, 25, (3, 3), 1, 0),
        nn.MaxPool2d((3, 3), 2),
        *conv(25, 50, (3, 3), 1, 0),  # 50*27*27
        *conv(50, 75, (3, 3), 1, 0),  # 75*25*25
        nn.MaxPool2d((3, 3), 2),  # 75*12*12
        Lambda(flatten),
        nn.Linear(10800, 2500),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(2500, 250),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(250, 5 * 5),
        Lambda(clamp),
        Lambda(round_),
    )
    return arch
