import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from math import sqrt
from pathlib import Path


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def reshape(x):
    return x.reshape(-1,2,63,63)

def normalization(x):
    """
    Normalize each Image to have amplitudes elements of [0,1]
    [min,max] -> [0,1]
    """
    norm = ()
    for j in range(len(x[:,0,0])):
        b = torch.max(x[j]).item()
        a = torch.min(x[j]).item()
        n = (x[j] - a)/(b-a)
        norm = norm + (n,)
    return torch.stack(norm)

def fft(x):
    """
    Layer that performs a fast Fourier-Transformation.
    """
    img_size = x.size(1) // 2
    # sort the incoming tensor in real and imaginary part
    arr_real = x[:, 0:img_size].reshape(-1, int(sqrt(img_size)), int(sqrt(img_size)))
    arr_imag = x[:, img_size:].reshape(-1, int(sqrt(img_size)), int(sqrt(img_size)))
    arr = torch.stack((arr_real, arr_imag), dim=-1)
    # perform fourier transformation and switch imaginary and real part
    arr_fft = torch.ifft(arr, 2).permute(0, 3, 2, 1).transpose(2, 3)
    # shift the lower frequencies in the middle
    # axes = tuple(range(arr_fft.ndim))
    # shift = [-(dim // 2) for dim in arr_fft.shape]
    # arr_shift = torch.roll(arr_fft, shift, axes)
    return arr_fft

def fft_(x):
    """
    Layer that performs an inverse fast Fourier_Transformation on a non 
    flattened Tensor. First dimension should be the number of the image.
    Second should be if real or imaginary. Best if x = euler_(y)
    """
    X = x.permute(0,2,3,1)
    X = torch.ifft(X,2)

    arr = (X[:,:,:,0]**2 + X[:,:,:,1]**2)**0.5
    return arr

def unsqueeze1(x):
    return x.unsqueeze(1)

def shape(x):
    print(x.shape)
    return x

def absolute(x):
    return torch.abs(x)

def euler(x):
    img_size = x.size(1) // 2
    arr_amp = x[:, 0:img_size]
    arr_phase = x[:, img_size:]

    arr_real = arr_amp * torch.cos(arr_phase)
    arr_imag = arr_amp * torch.sin(arr_phase)

    arr = torch.stack((arr_real, arr_imag), dim=-1).permute(0, 2, 1)
    return arr

def euler_(x):
    """
    Same as euler, but for non flattened Tensors.
    """
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    amp = x[:,0]
    pha = x[:,1]

    real = amp * torch.cos(pha)
    imag = amp * torch.sin(pha)

    arr = torch.stack((real, imag), dim=1)
    return arr

def flatten(x):
    return x.reshape(x.shape[0], -1)


def flatten_with_channel(x):
    return x.reshape(x.shape[0], x.shape[1], -1)


def cut_off(x):
    a = x.clone()
    a[a <= 1e-10] = 1e-10
    return a


def symmetry(x, mode="real"):
    center = (x.shape[1]) // 2
    u = torch.arange(center)
    v = torch.arange(center)

    diag1 = torch.arange(center, x.shape[1])
    diag2 = torch.arange(center, x.shape[1])
    diag_indices = torch.stack((diag1, diag2))
    grid = torch.tril_indices(x.shape[1], x.shape[1], -1)

    x_sym = torch.cat((grid[0].reshape(-1, 1), diag_indices[0].reshape(-1, 1)),)
    y_sym = torch.cat((grid[1].reshape(-1, 1), diag_indices[1].reshape(-1, 1)),)
    x = torch.rot90(x, 1, dims=(1, 2))
    i = center + (center - x_sym)
    j = center + (center - y_sym)
    u = center - (center - x_sym)
    v = center - (center - y_sym)
    if mode == "real":
        x[:, i, j] = x[:, u, v]
    if mode == "imag":
        x[:, i, j] = -x[:, u, v]
    return torch.rot90(x, 3, dims=(1, 2))


class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak = leak
        self.sub = sub
        self.maxv = maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x.sub_(self.sub)
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x


class GeneralELU(nn.Module):
    def __init__(
        self, add=None,
    ):
        super().__init__()
        self.add = add

    def forward(self, x):
        x = F.elu(x)
        if self.add is not None:
            x = x + self.add
        return x


def init_cnn_(m, f):
    if isinstance(m, nn.Conv2d):
        f(m.weight, a=0.1)
        if getattr(m, "bias", None) is not None:
            m.bias.data.zero_()
    for l in m.children():
        init_cnn_(l, f)


def init_cnn(m, uniform=False):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    init_cnn_(m, f)


def conv_layer(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [
        nn.Conv2d(ni, nf, ks, padding=ks // 2, stride=stride, bias=not bn),
        GeneralRelu(**kwargs),
    ]
    if bn:
        layers.append(nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1))
    return nn.Sequential(*layers)


class RunningBatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones(nf, 1, 1))
        self.adds = nn.Parameter(torch.zeros(nf, 1, 1))
        self.register_buffer("sums", torch.zeros(1, nf, 1, 1))
        self.register_buffer("sqrs", torch.zeros(1, nf, 1, 1))
        self.register_buffer("count", torch.tensor(0.0))
        self.register_buffer("factor", torch.tensor(0.0))
        self.register_buffer("offset", torch.tensor(0.0))
        self.batch = 0

    def update_stats(self, x):
        bs, nc, *_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0, 2, 3)
        s = x.sum(dims, keepdim=True)
        ss = (x * x).sum(dims, keepdim=True)
        c = s.new_tensor(x.numel() / nc)
        mom1 = s.new_tensor(1 - (1 - self.mom) / sqrt(bs - 1))
        self.sums.lerp_(s, mom1)
        self.sqrs.lerp_(ss, mom1)
        self.count.lerp_(c, mom1)
        self.batch += bs
        means = self.sums / self.count
        varns = (self.sqrs / self.count).sub_(means * means)
        if bool(self.batch < 20):
            varns.clamp_min_(0.01)
        self.factor = self.mults / (varns + self.eps).sqrt()
        self.offset = self.adds - means * self.factor

    def forward(self, x):
        if self.training:
            self.update_stats(x)
        return x * self.factor + self.offset


def conv(ni, nc, ks, stride, padding):
    conv = (nn.Conv2d(ni, nc, ks, stride, padding),)
    bn = (nn.BatchNorm2d(nc),)
    act = GeneralRelu(leak=0.1, sub=0.4)  # nn.ReLU()
    layers = [*conv, *bn, act]
    return layers


def conv_amp(ni, nc, ks, stride, padding, dilation):
    """Create a convolutional layer for the amplitude reconstruction.
    The activation function ist ReLU with a 2d Batch normalization.

    Parameters
    ----------
    ni : int
        Number of input channels
    nc : int
        Number of output channels
    ks : tuple
        Size of the kernel
    stride : int
        Stepsize between use of kernel
    padding : int
        Number of pixels added to edges of picture
    dilation : int
        Factor for spreading the receptive field

    Returns
    -------
    list
        list of convolutional layer, 2d Batch Normalisation and Activation function.
    """
    conv = (
        nn.Conv2d(
            ni, nc, ks, stride, padding, dilation, bias=False, padding_mode="replicate"
        ),
    )
    bn = (nn.BatchNorm2d(nc),)
    act = nn.ReLU()
    layers = [*conv, *bn, act]
    return layers


def conv_phase(ni, nc, ks, stride, padding, dilation, add):
    """Create a convolutional layer for the amplitude reconstruction.
    The activation function ist GeneralELU with a 2d Batch normalization.

    Parameters
    ----------
    ni : int
        Number of input channels
    nc : int
        Number of output channels
    ks : tuple
        Size of the kernel
    stride : int
        Stepsize between use of kernel
    padding : int
        Number of pixels added to edges of picture
    dilation : int
        Factor for spreading the receptive field
    add : int
        Number which is added to GeneralELU

    Returns
    -------
    list
        list of convolutional layer, 2d Batch Normalisation and Activation function.
    """
    conv = (
        nn.Conv2d(
            ni, nc, ks, stride, padding, dilation, bias=False, padding_mode="replicate"
        ),
    )
    bn = (nn.BatchNorm2d(nc),)
    act = GeneralELU(add)
    layers = [*conv, *bn, act]
    return layers


def depth_conv(ni, nc, ks, stride, padding, dilation):
    conv = (nn.Conv2d(ni, nc, ks, stride, padding, dilation=dilation, groups=ni),)
    bn = (nn.BatchNorm2d(nc),)
    act = GeneralRelu(leak=0.1, sub=0.4)  # nn.ReLU()
    layers = [*conv, *bn, act]
    return layers


def double_conv(ni, nc, ks=3, stride=1, padding=1):
    conv = (nn.Conv2d(ni, nc, ks, stride, padding),)
    bn = (nn.BatchNorm2d(nc),)
    act = (nn.ReLU(inplace=True),)
    conv2 = (nn.Conv2d(nc, nc, ks, stride, padding),)
    bn2 = (nn.BatchNorm2d(nc),)
    act2 = nn.ReLU(inplace=True)
    layers = [*conv, *bn, *act, *conv2, *bn2, act2]
    return layers


def deconv(ni, nc, ks, stride, padding, out_padding):
    conv = (nn.ConvTranspose2d(ni, nc, ks, stride, padding, out_padding),)
    bn = (nn.BatchNorm2d(nc),)
    act = GeneralRelu(leak=0.1, sub=0.4)  # nn.ReLU()
    layers = [*conv, *bn, act]
    return layers


def load_pre_model(learn, pre_path, visualize=False, lr_find=False):
    """
    :param learn:       object of type learner
    :param pre_path:    string wich contains the path of the model
    :param lr_find:     bool which is True if lr_find is used
    """
    name_pretrained = pre_path.split("/")[-1].split(".")[0]
    print("\nLoad pretrained model: {}\n".format(name_pretrained))

    if visualize:
        checkpoint = torch.load(pre_path)
        learn.load_state_dict(checkpoint["model_state_dict"])

    else:
        checkpoint = torch.load(pre_path)
        learn.model.load_state_dict(checkpoint["model_state_dict"])
        learn.opt = learn.opt_func(learn.model.parameters(), learn.lr).load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        learn.epoch = checkpoint["epoch"]
        learn.loss = checkpoint["loss"]
        if not lr_find:
            learn.recorder.train_losses = checkpoint["recorder_train_loss"]
            learn.recorder.valid_losses = checkpoint["recorder_valid_loss"]
            learn.recorder.losses = checkpoint["recorder_losses"]
            learn.recorder.lrs = checkpoint["recorder_lrs"]
        else:
            learn.recorder_lr_find.losses = checkpoint["recorder_losses"]
            learn.recorder_lr_find.lrs = checkpoint["recorder_lrs"]


def save_model(learn, model_path):
    state = learn.model.state_dict()
    p = Path(model_path).parent
    p.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": learn.epoch,
            "model_state_dict": state,
            "optimizer_state_dict": learn.opt.state_dict(),
            "loss": learn.loss,
            "recorder_train_loss": learn.recorder.train_losses,
            "recorder_valid_loss": learn.recorder.valid_losses,
            "recorder_losses": learn.recorder.losses,
            "recorder_lrs": learn.recorder.lrs,
        },
        model_path,
    )


class LocallyConnected2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, output_size, kernel_size, stride, bias=False
    ):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(
                1,
                out_channels,
                in_channels,
                output_size[0],
                output_size[1],
                kernel_size ** 2,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter("bias", None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out
