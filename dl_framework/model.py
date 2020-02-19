import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


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


def shape(x):
    print(x.shape)
    return x


def flatten(x):
    return x.reshape(x.shape[0], -1)


def cut_off(x):
    a = x.clone()
    a[a <= 1e-10] = 1e-10
    return a


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


def init_cnn_(m, f):
    if isinstance(m, nn.Conv2d):
        f(m.weight, a=0.1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()
    for l in m.children():
        init_cnn_(l, f)


def init_cnn(m, uniform=False):
    f = nn.init.kaiming_uniform_ if uniform else nn.init.kaiming_normal_
    init_cnn_(m, f)


def conv_layer(ni, nf, ks=3, stride=2, bn=True, **kwargs):
    layers = [nn.Conv2d(ni, nf, ks, padding=ks//2, stride=stride, bias=not bn),
              GeneralRelu(**kwargs)]
    if bn:
        layers.append(nn.BatchNorm2d(nf, eps=1e-5, momentum=0.1))
    return nn.Sequential(*layers)


class RunningBatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom, self.eps = mom, eps
        self.mults = nn.Parameter(torch.ones(nf, 1, 1))
        self.adds = nn.Parameter(torch.zeros(nf, 1, 1))
        self.register_buffer('sums', torch.zeros(1, nf, 1, 1))
        self.register_buffer('sqrs', torch.zeros(1, nf, 1, 1))
        self.register_buffer('count', torch.tensor(0.))
        self.register_buffer('factor', torch.tensor(0.))
        self.register_buffer('offset', torch.tensor(0.))
        self.batch = 0

    def update_stats(self, x):
        bs, nc, *_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0, 2, 3)
        s = x.sum(dims, keepdim=True)
        ss = (x*x).sum(dims, keepdim=True)
        c = s.new_tensor(x.numel()/nc)
        mom1 = s.new_tensor(1 - (1-self.mom)/sqrt(bs-1))
        self.sums .lerp_(s, mom1)
        self.sqrs .lerp_(ss, mom1)
        self.count.lerp_(c, mom1)
        self.batch += bs
        means = self.sums/self.count
        varns = (self.sqrs/self.count).sub_(means*means)
        if bool(self.batch < 20):
            varns.clamp_min_(0.01)
        self.factor = self.mults / (varns+self.eps).sqrt()
        self.offset = self.adds - means*self.factor

    def forward(self, x):
        if self.training:
            self.update_stats(x)
        return x*self.factor + self.offset


def conv(ni, nc, ks, stride, padding):
    conv = nn.Conv2d(ni, nc, ks, stride, padding),
    bn = nn.BatchNorm2d(nc),
    act = GeneralRelu(leak=0.1, sub=0.4)  # nn.ReLU()
    layers = [*conv, *bn, act]
    return layers


def double_conv(ni, nc, ks, stride, padding):
    conv = nn.Conv2d(ni, nc, ks, stride, padding),
    bn = nn.BatchNorm2d(nc),
    act = nn.ReLU(inplace=True),
    conv2 = nn.Conv2d(nc, nc, ks, stride, padding),
    bn2 = nn.BatchNorm2d(nc),
    act2 = nn.ReLU(inplace=True)
    layers = [*conv, *bn, *act, *conv2, *bn2, act2]
    return layers


def deconv(ni, nc, ks, stride, padding, out_padding):
    conv = nn.ConvTranspose2d(ni, nc, ks, stride, padding, out_padding),
    bn = nn.BatchNorm2d(nc),
    act = GeneralRelu(leak=0.1, sub=0.4)  # nn.ReLU()
    layers = [*conv, *bn, act]
    return layers

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def load_pre_model(learn, pre_path, visualize=False, lr_find=False):
    """
    :param learn:       object of type learner
    :param pre_path:    string wich contains the path of the model
    :param lr_find:     bool which is True if lr_find is used
    """
    name_pretrained = pre_path.split("/")[-1].split(".")[0]
    print('\nLoad pretrained model: {}\n'.format(name_pretrained))

    if visualize:
        checkpoint = torch.load(pre_path)
        learn.load_state_dict(checkpoint['model_state_dict'])

    else:
        checkpoint = torch.load(pre_path)
        learn.model.load_state_dict(checkpoint['model_state_dict'])
        learn.opt = learn.opt_func(learn.model.parameters(), learn.lr).load_state_dict(checkpoint['optimizer_state_dict'])
        learn.epoch = checkpoint['epoch']
        learn.loss = checkpoint['loss']
        if not lr_find:
            learn.recorder.train_losses = checkpoint['recorder_train_loss']
            learn.recorder.valid_losses = checkpoint['recorder_valid_loss']
            learn.recorder.losses = checkpoint['recorder_losses']
            learn.recorder.lrs = checkpoint['recorder_lrs']
        else:
            learn.recorder_lr_find.losses = checkpoint['recorder_losses']
            learn.recorder_lr_find.lrs = checkpoint['recorder_lrs']


def save_model(learn, model_path):
    state = learn.model.state_dict()
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
