from torch import nn
from dl_framework.model import (
    conv,
    Lambda,
    flatten,
    unsqueeze1,
    absolute,
    normalization,
    clamp,
    round_,
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
        nn.Linear(470, 5 * 2),
        Lambda(clamp),
        #Lambda(round_),
    )
    return arch


def list_pos():
    """
    Conv-Network with source list as target. Position only. FFT beforehand.
    """
    arch = nn.Sequential(
        Lambda(normalization),
        Lambda(unsqueeze1),
        *conv(1, 3, (3, 3), 1, 1),  # 3*63*63
        *conv(3, 6, (3, 3), 1, 1),  # 6*63*63
        Lambda(flatten),
        nn.Linear(23814, 2300),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(2300, 230),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(230, 5 * 2),
        Lambda(clamp),
    )
    return arch


def onesource():
    """
    For one source images.
    """
    arch = nn.Sequential(
        Lambda(unsqueeze1),
        *conv(1, 5, (3, 3), 1, 1),
        Lambda(flatten),
        nn.Linear(19845, 5000),
        nn.ReLU(),
        nn.Linear(5000, 500),
        nn.ReLU(),
        nn.Linear(500, 5),
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
