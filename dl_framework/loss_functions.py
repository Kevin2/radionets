import torch
from torch import nn
from dl_framework.hook_fastai import hook_outputs
from torchvision.models import vgg16_bn
from dl_framework.utils import children
import torch.nn.functional as F
import pytorch_msssim


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, base_loss, layer_ids, layer_wgts):
        """"
        m_feat: enthält das vortrainierte Netz
        loss_features: dort werden alle features gespeichert, deren Loss
        man berechnen will
        """
        super().__init__()
        self.m_feat = m_feat
        self.base_loss = base_loss
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        # self.metric_names = (
        #     ["pixel", ]
        #     + [f"feat_{i}" for i in range(len(layer_ids))]
        #     + [f"gram_{i}" for i in range(len(layer_ids))]
        # )

    def make_features(self, x, clone=False):
        """"
        Hier wird das Objekt x durch das vortrainierte Netz geschickt und somit die
        Aktivierungsfunktionen berechnet. Dies geschieht sowohl einmal für
        die Wahrheit "target" und einmal für die Prediction "input"
        aus dem Generator. Dann werden die berechneten Aktivierungsfunktionen als Liste
        gespeichert und zurückgegeben.
        """
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        # resizing the input, before it gets into the net
        # shape changes from 4096 to 64x64
        target = target.view(-1, 2, 64, 64)
        input = input.view(-1, 2, 64, 64)

        # create dummy tensor of zeros to add another dimension
        padding_target = torch.zeros(
            target.size(0), 1, target.size(2), target.size(3)
        ).cuda()
        padding_input = torch.zeros(
            input.size(0), 1, input.size(2), input.size(3)
        ).cuda()

        # 'add' the extra channel
        target = torch.cat((target, padding_target), 1)
        input = torch.cat((input, padding_input), 1)

        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)

        # Hier wird jetzt der L1-Loss zwischen Input und Target berechnet
        self.feat_losses = [self.base_loss(input, target)]

        # hier wird das gleiche nochmal für alle Features gemacht
        self.feat_losses += [
            self.base_loss(f_in, f_out)
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        # erstmal den Teil mit der gram_matrix auskommentiert, bis er
        # verstanden ist
        self.feat_losses += [
            self.base_loss(gram_matrix(f_in), gram_matrix(f_out))
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        # Wird als Liste gespeichert, um es in metrics abspeichern
        # zu können und printen zu können

        # erstmal unnötig
        # self.metrics = dict(zip(self.metric_names, self.feat_losses))

        # zum Schluss wird hier aufsummiert
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2)) / (c * h * w)


def init_feature_loss(
    pre_net=vgg16_bn,
    pixel_loss=F.l1_loss,
    begin_block=2,
    end_block=5,
    layer_weights=[5, 15, 2],
):
    """
    method to initialise  the pretrained net, which will be used for the feature loss.
    """
    vgg_m = pre_net(True).features.cuda().eval()
    for param in vgg_m.parameters():
        param.requires_grad = False
    blocks = [
        i - 1 for i, o in enumerate(children(vgg_m)) if isinstance(o, nn.MaxPool2d)
    ]
    feat_loss = FeatureLoss(
        vgg_m, pixel_loss, blocks[begin_block:end_block], layer_weights
    )
    return feat_loss


def splitted_mse(x, y):
    inp_real = x[:, 0, :]
    inp_imag = x[:, 1, :]

    tar_real = y[:, 0, :]
    tar_imag = y[:, 1, :]

    loss_real = (
        torch.sum(1 / inp_real.shape[1] * torch.sum((inp_real - tar_real) ** 2, 1))
        * 1
        / inp_real.shape[0]
    )
    loss_imag = (
        torch.sum(1 / inp_imag.shape[1] * torch.sum((inp_imag - tar_imag) ** 2, 1))
        * 1
        / inp_real.shape[0]
    )
    print(inp_real.shape)
    print(tar_real.shape)

    return loss_real + loss_imag


def loss_amp(x, y):
    tar = y[:, 0, :].unsqueeze(1)
    assert tar.shape == x.shape

    mse = nn.MSELoss()
    loss = mse(x, tar)

    return loss


def loss_phase(x, y):
    tar = y[:, 1, :].unsqueeze(1)
    assert tar.shape == x.shape

    mse = nn.MSELoss()
    loss = mse(x, tar)

    return loss


def loss_msssim(x, y):
    """Loss function with 1 - the value of the MS-SSIM

    Parameters
    ----------
    x : tensor
        output of net
    y : tensor
        output of net

    Returns
    -------
    float
        value of 1 - MS-SSIM
    """
    inp_real = x[:, 0, :].unsqueeze(1)
    inp_imag = x[:, 1, :].unsqueeze(1)

    tar_real = y[:, 0, :].unsqueeze(1)
    tar_imag = y[:, 1, :].unsqueeze(1)

    loss = (
        1.0
        - pytorch_msssim.msssim(inp_real, tar_real, normalize="relu")
        + 1.0
        - pytorch_msssim.msssim(inp_imag, tar_imag, normalize="relu")
    )

    return loss


def loss_msssim_diff(x, y):
    """Loss function with negative value of MS-SSIM

    Parameters
    ----------
    x : tensor
        output of net
    y : tensor
        target image

    Returns
    -------
    float
        value of negative MS-SSIM
    """
    inp_real = x[:, 0, :].unsqueeze(1)
    inp_imag = x[:, 1, :].unsqueeze(1)

    tar_real = y[:, 0, :].unsqueeze(1)
    tar_imag = y[:, 1, :].unsqueeze(1)

    loss = -(
        pytorch_msssim.msssim(inp_real, tar_real, normalize="relu")
        + pytorch_msssim.msssim(inp_imag, tar_imag, normalize="relu")
    )

    return loss


def loss_msssim_amp(x, y):
    """Loss function with 1 - the value of the MS-SSIM for amplitude

    Parameters
    ----------
    x : tensor
        output of net
    y : tensor
        output of net

    Returns
    -------
    float
        value of 1 - MS-SSIM
    """
    inp_real = x
    tar_real = y[:, 0, :].unsqueeze(1)

    loss = (
        1.0
        - pytorch_msssim.msssim(inp_real, tar_real, normalize="relu")
    )

    return loss


def loss_mse_msssim_phase(x, y):
    """Combine MSE and MS-SSIM loss for phase

    Parameters
    ----------
    x : tensor
        ouptut of net
    y : tensor
        target image

    Returns
    -------
    float
        value of addition of MSE and MS-SSIM
    """

    tar_phase = y[:, 1, :].unsqueeze(1)
    inp_phase = x

    loss_mse_phase = nn.MSELoss()
    loss_mse_phase = loss_mse_phase(inp_phase, tar_phase)

    loss_phase = (
        loss_mse_phase
        + 1.0
        - pytorch_msssim.msssim(inp_phase, tar_phase, normalize="relu")
    )

    return loss_phase


def loss_mse_msssim_amp(x, y):
    """Combine MSE and MS-SSIM loss for amp

    Parameters
    ----------
    x : tensor
        ouptut of net
    y : tensor
        target image

    Returns
    -------
    float
        value of addition of MSE and MS-SSIM
    """

    tar_amp = y[:, 0, :].unsqueeze(1)
    inp_amp = x

    loss_mse_amp = nn.MSELoss()
    loss_mse_amp = loss_mse_amp(inp_amp, tar_amp)

    loss_amp = (
        loss_mse_amp + 1.0 - pytorch_msssim.msssim(inp_amp, tar_amp, normalize="relu")
    )

    return loss_amp


def loss_mse_msssim(x, y):
    """Combine MSE and MS-SSIM loss

    Parameters
    ----------
    x : tensor
        output of net
    y : tensor
        target image

    Returns
    -------
    float
        value of addition of MSE and MS-SSIM
    """
    # Split in amplitude and phase
    inp_amp = x[:, 0, :].unsqueeze(1)
    inp_phase = x[:, 1, :].unsqueeze(1)

    tar_amp = y[:, 0, :].unsqueeze(1)
    tar_phase = y[:, 1, :].unsqueeze(1)

    # calculate mse loss
    loss_mse_amp = nn.MSELoss()
    loss_mse_amp = loss_mse_amp(inp_amp, tar_amp)

    loss_mse_phase = nn.MSELoss()
    loss_mse_phase = loss_mse_phase(inp_phase, tar_phase)

    # add mse loss to MS-SSIM
    loss_amp = (
        loss_mse_amp + 1.0 - pytorch_msssim.msssim(inp_amp, tar_amp, normalize="relu")
    )
    loss_phase = (
        loss_mse_phase
        + 1.0
        - pytorch_msssim.msssim(inp_phase, tar_phase, normalize="relu")
    )

    return loss_amp + loss_phase

def list_loss(x, y):
    """
    Adapted loss for source list output.
    """
    x = x.reshape(-1, 5, 5)
    inp_pos = x[:,:,:2]
    inp_wdth = x[:,:,2:4]
    inp_amp = x[:,:,4]

    tar_pos = y[:,:,:2]
    tar_wdth = y[:,:,2:4]
    tar_amp = y[:,:,4]

    loss_pos = nn.SmoothL1Loss()
    loss_pos = loss_pos(inp_pos, tar_pos)

    loss_amp_wdth = nn.MSELoss()
    loss_amp = loss_amp_wdth(inp_amp, tar_amp)
    loss_wdth = loss_amp_wdth(inp_wdth, tar_wdth)

    return loss_pos + loss_amp + loss_wdth
