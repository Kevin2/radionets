import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from math import sqrt, pi
from fastcore.foundation import L


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def permutation(x, y, out_loss=False):
    """
    Returns the permutation minimizing Mean Squared Error
    between predictions x and target y using hungarian
    algorithm or Mean Squared Error.

    Input
    ------
    x : torch tensor (:,5,2)
        Predicted Source positions
    y : torch tensor (:,5,2)
        True Source positions
    out_loss: bool
        Decide between output perm or loss

    Output
    ------
    perm: np-array
        Permutation (for x) which minimizes euclidean distance.
    or
    loss: float
        Mean Squared Error on best permutation
    """
    d = nn.MSELoss(reduction="mean")
    cost = torch.empty((x.shape[0], y.shape[0]))
    for i in range(len(cost)):
        for j in range(len(cost)):
            cost[i, j] = d(x[i], y[j])
    row, col = linear_sum_assignment(cost)
    perm = np.zeros(col.shape)
    for j in range(len(col)):
        perm[col[j]] = j
    loss = cost[row, col].mean()
    if out_loss is False:
        return perm
    else:
        return loss


def sort(x, entry=0):
    a = x[:, entry]
    if type(a) == np.ndarray:
        a = torch.from_numpy(a)
    _, indices = torch.sort(a)
    x = x[indices, :]
    return x


def reshape(x):
    return x.reshape(-1, 2, 63, 63)


def compress_image(img):
    part1, part2, part3, part4 = split_parts(img, pad=True)

    part1_1, _, part1_3, _ = split_parts(part1, pad=False)
    part2_1, part2_2, part2_3, part2_4 = split_parts(part2, pad=False)
    part3_1, part3_2, _, _ = split_parts(part3, pad=False)
    part4_1, part4_2, _, _ = split_parts(part4, pad=False)

    return (
        part1_1,
        part1_3,
        part2_1,
        part2_2,
        part2_3,
        part2_4,
        part3_1,
        part3_2,
        part4_1,
        part4_2,
    )


def expand_image(params):
    (
        part1_1,
        part1_3,
        part2_1,
        part2_2,
        part2_3,
        part2_4,
        part3_1,
        part3_2,
        part4_1,
        part4_2,
    ) = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
        params[6],
        params[7],
        params[8],
        params[9],
    )
    bs = part1_1.shape[0]
    part1 = combine_parts(
        [
            part1_1,
            -torch.rot90(part1_1, 2, dims=(2, 3)),
            part1_3,
            -torch.rot90(part1_3, 2, dims=(2, 3)),
            (bs, 1, 32, 32),
            False,
        ]
    )

    part2 = combine_parts([part2_1, part2_2, part2_3, part2_4, (bs, 1, 32, 32), False])

    part3 = combine_parts(
        [
            part3_1,
            part3_2,
            F.pad(
                input=-torch.rot90(part3_2[:, :, :, :-1], 2, dims=(2, 3)),
                pad=(0, 1, 0, 0),
                mode="constant",
                value=0,
            ),
            -torch.rot90(part3_1, 2, dims=(2, 3)),
            (bs, 1, 32, 32),
            False,
        ]
    )

    part4 = combine_parts(
        [
            part4_1,
            part4_2,
            -torch.rot90(part4_1, 2, dims=(2, 3)),
            F.pad(
                input=-torch.rot90(part4_2[:, :, :-1, :], 2, dims=(2, 3)),
                pad=(0, 0, 0, 1),
                mode="constant",
                value=0,
            ),
            (bs, 1, 32, 32),
            False,
        ]
    )

    img = combine_parts([part1, part2, part3, part4, (bs, 1, 63, 63), True])
    return img


def split_parts(img, pad=True):
    t_img = img.clone()
    part1 = t_img[:, 0, 0::2, 0::2]
    part2 = t_img[:, 0, 1::2, 1::2]
    part3 = t_img[:, 0, 0::2, 1::2]
    part4 = t_img[:, 0, 1::2, 0::2]
    if pad:
        # print("Padding done.")
        part2 = F.pad(input=part2, pad=(0, 1, 0, 1), mode="constant", value=0)
        part3 = F.pad(input=part3, pad=(0, 1, 0, 0), mode="constant", value=0)
        part4 = F.pad(input=part4, pad=(0, 0, 0, 1), mode="constant", value=0)
    return (
        part1.unsqueeze(1),
        part2.unsqueeze(1),
        part3.unsqueeze(1),
        part4.unsqueeze(1),
    )


def combine_parts(params):
    part1, part2, part3, part4, img_size, final = (
        params[0],
        params[1],
        params[2],
        params[3],
        params[4],
        params[5],
    )
    comb = torch.zeros(img_size).cuda()
    if final is True:
        comb[:, :, 0::2, 0::2] = part1
        comb[:, :, 1::2, 1::2] = part2[:, :, :-1, :-1]
        comb[:, :, 0::2, 1::2] = part3[:, :, :, :-1]
        comb[:, :, 1::2, 0::2] = part4[:, :, :-1, :]
    else:
        comb[:, :, 0::2, 0::2] = part1
        comb[:, :, 1::2, 1::2] = part2
        comb[:, :, 0::2, 1::2] = part3
        comb[:, :, 1::2, 0::2] = part4
    return comb


def round_(x):
    return torch.round(x)


def clamp(x):
    return torch.clamp(x, 0, 63)


def normalization(x):
    """
    Normalize each Image to have amplitudes elements of [0,1]
    [min,max] -> [0,1]
    """
    norm = ()
    for j in range(len(x[:, 0, 0])):
        b = torch.max(x[j]).item()
        a = torch.min(x[j]).item()
        n = (x[j] - a) / (b - a)
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
    X = x.permute(0, 2, 3, 1)
    X = torch.ifft(X, 2)

    arr = (X[:, :, :, 0] ** 2 + X[:, :, :, 1] ** 2) ** 0.5
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

    arr_real = (10 ** (10 * (arr_amp - 1)) - 1e-10) * torch.cos(arr_phase)
    arr_imag = (10 ** (10 * (arr_amp - 1)) - 1e-10) * torch.sin(arr_phase)

    arr = torch.stack((arr_real, arr_imag), dim=-1).permute(0, 2, 1)
    return arr


def euler_(x):
    """
    Same as euler, but for non flattened Tensors.
    """
    if type(x) == np.ndarray:
        x = torch.from_numpy(x)
    amp = x[:, 0]
    pha = x[:, 1]

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

    x_sym = torch.cat(
        (grid[0].reshape(-1, 1), diag_indices[0].reshape(-1, 1)),
    )
    y_sym = torch.cat(
        (grid[1].reshape(-1, 1), diag_indices[1].reshape(-1, 1)),
    )
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


def phase_range(phase):
    # if isinstance(phase, float):
    #     phase = torch.tensor([phase])
    mult = phase / pi
    mult[mult <= 1] = 0
    mult[mult % 2 <= 1] -= 1
    mult = torch.round(mult / 2)
    mult[(phase / pi > 1) & (mult == 0)] = 1
    phase = phase - mult * 2 * pi
    return phase


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
    def __init__(self, add=None, maxv=None):
        super().__init__()
        self.add = add
        self.maxv = maxv

    def forward(self, x):
        x = F.elu(x)
        if self.add is not None:
            x = x + self.add
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
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


def load_pre_model(learn, pre_path, visualize=False):
    """
    :param learn:       object of type learner
    :param pre_path:    string wich contains the path of the model
    :param lr_find:     bool which is True if lr_find is used
    """
    name_pretrained = Path(pre_path).stem
    print("\nLoad pretrained model: {}\n".format(name_pretrained))

    if visualize:
        checkpoint = torch.load(pre_path)
        learn.load_state_dict(checkpoint["model_state_dict"])

    else:
        checkpoint = torch.load(pre_path)
        learn.model.load_state_dict(checkpoint["model"])
        learn.opt.load_state_dict(checkpoint["opt"])
        learn.epoch = checkpoint["epoch"]
        learn.loss = checkpoint["loss"]
        learn.recorder.iters = checkpoint["iters"]
        learn.recorder.values = checkpoint["vals"]
        learn.recorder.train_losses = checkpoint["recorder_train_loss"]
        learn.recorder.valid_losses = checkpoint["recorder_valid_loss"]
        learn.recorder.losses = checkpoint["recorder_losses"]
        learn.recorder.lrs = checkpoint["recorder_lrs"]


def save_model(learn, model_path):
    torch.save(
        {
            "model": learn.model.state_dict(),
            "opt": learn.opt.state_dict(),
            "epoch": learn.epoch,
            "loss": learn.loss,
            "iters": learn.recorder.iters,
            "vals": learn.recorder.values,
            "recorder_train_loss": L(learn.recorder.values[0:]).itemgot(0),
            "recorder_valid_loss": L(learn.recorder.values[0:]).itemgot(1),
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


def create_rot_mat(alpha):
    rot_mat = torch.tensor(
        [[torch.cos(alpha), -torch.sin(alpha)], [torch.sin(alpha), torch.cos(alpha)]]
    )
    rot_mat = rot_mat.cuda()
    return rot_mat


def gaussian_component(x, y, flux, x_fwhm, y_fwhm, rot, center=None):
    if center is None:
        x_0 = y_0 = len(x) // 2
    else:
        rot_mat = create_rot_mat(torch.deg2rad(rot))
        x_0, y_0 = ((center - len(x) // 2) @ rot_mat) + len(x) // 2
    gauss = flux * torch.exp(
        -((x_0 - x) ** 2 / (2 * (x_fwhm) ** 2) + (y_0 - y) ** 2 / (2 * (y_fwhm) ** 2))
    )
    gauss = gauss.cuda()
    return gauss


def create_grid(pixel):
    x = torch.linspace(0, pixel - 1, steps=pixel)
    y = torch.linspace(0, pixel - 1, steps=pixel)
    X, Y = torch.meshgrid(x, y)
    X = X.cuda()
    Y = Y.cuda()
    X.unsqueeze_(0)
    Y.unsqueeze_(0)
    mesh = torch.cat((X, Y))
    x = torch.zeros(X.shape) + 1e-10
    # grid = torch.tensor((x))
    grid = x.clone().detach()
    grid = grid.cuda()
    grid = torch.cat((grid, mesh))
    return grid


def gauss_valid(params):  # setzt aus den einzelen parametern (54) ein bild zusammen
    gauss_param = torch.split(params, 9)
    grid = create_grid(63)
    source = torch.tensor((grid[0]))
    source = grid.clone().detach()
    for i in range(len(gauss_param)):
        cent = torch.tensor(
            [
                len(grid[0]) // 2 + gauss_param[1][i],
                len(grid[0]) // 2 + gauss_param[2][i],
            ]
        )
        cent = cent.cuda()
        s = gaussian_component(
            grid[1],
            grid[2],
            gauss_param[0][i],
            gauss_param[3][i],
            gauss_param[4][i],
            rot=gauss_param[5][i],
            center=cent,
        )
        source = torch.add(source, s)
    return source


def vaild_gauss_bs(in_put):
    for i in range(in_put.shape[0]):
        if i == 0:
            source = gauss_valid(in_put[i])  # gauss parameter des ersten gausses
            source.unsqueeze_(0)
        else:
            h = gauss_valid(in_put[i])
            h.unsqueeze_(0)
            source = torch.cat((source, h))
    return source
