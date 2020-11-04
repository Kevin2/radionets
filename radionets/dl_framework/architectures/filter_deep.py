from torch import nn
import torch
from radionets.dl_framework.model import (
    Lambda,
    flatten,
    fft,
    LocallyConnected2d,
    symmetry,
    shape,
    GeneralELU,
    conv_phase,
    conv_amp,
    euler,
)
from functools import partial
from math import pi

from radionets.dl_framework.uncertainty_arch import (
    block_1_a,
    block_2_a,
    block_3_a,
    block_1_a_unc,
    block_2_a_unc,
    block_3_a_unc,
)

from radionets.dl_framework.uncertainty_arch import (
    block_1_p,
    block_2_p,
    block_3_p,
    block_4_p,
    bridge,
    block_1_p_unc,
    block_2_p_unc,
    block_3_p_unc,
)


class filter_deep_amp(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.conv1_amp = nn.Sequential(*conv_amp(1, 4, (23, 23), 1, 11, 1))
        self.conv2_amp = nn.Sequential(*conv_amp(4, 8, (21, 21), 1, 10, 1))
        self.conv3_amp = nn.Sequential(*conv_amp(8, 12, (17, 17), 1, 8, 1))
        self.conv_con1_amp = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.conv4_amp = nn.Sequential(*conv_amp(1, 4, (5, 5), 1, 4, 2))
        self.conv5_amp = nn.Sequential(*conv_amp(4, 8, (5, 5), 1, 2, 1))
        self.conv6_amp = nn.Sequential(*conv_amp(8, 12, (3, 3), 1, 2, 2))
        self.conv7_amp = nn.Sequential(*conv_amp(12, 16, (3, 3), 1, 1, 1))
        self.conv_con2_amp = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.conv8_amp = nn.Sequential(*conv_amp(1, 4, (3, 3), 1, 1, 1))
        self.conv9_amp = nn.Sequential(*conv_amp(4, 8, (3, 3), 1, 1, 1))
        self.conv10_amp = nn.Sequential(*conv_amp(8, 12, (3, 3), 1, 2, 2))
        self.conv_con3_amp = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.symmetry_real = Lambda(symmetry)

    def forward(self, x):
        x = x[:, 0].unsqueeze(1)
        inp = x.clone()
        amp = x[:, 0, :].unsqueeze(1)

        amp = self.conv1_amp(amp)

        amp = self.conv2_amp(amp)

        amp = self.conv3_amp(amp)

        amp = self.conv_con1_amp(amp)

        # Second block
        amp = self.conv4_amp(amp)

        amp = self.conv5_amp(amp)

        amp = self.conv6_amp(amp)

        amp = self.conv7_amp(amp)

        amp = self.conv_con2_amp(amp)

        # Third block
        amp = self.conv8_amp(amp)

        amp = self.conv9_amp(amp)

        amp = self.conv10_amp(amp)

        amp = self.conv_con3_amp(amp)

        inp_amp = inp[:, 0].unsqueeze(1)
        x0 = self.symmetry_real(amp).reshape(-1, 1, amp.shape[2], amp.shape[2])
        x0[inp_amp != 0] = inp_amp[inp_amp != 0]

        return x0


class filter_deep_phase(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.conv1_phase = nn.Sequential(
            *conv_phase(1, 4, (23, 23), 1, 11, 1, add=-2.1415)
        )
        self.conv2_phase = nn.Sequential(
            *conv_phase(4, 8, (21, 21), 1, 10, 1, add=-2.1415)
        )
        self.conv3_phase = nn.Sequential(
            *conv_phase(8, 12, (17, 17), 1, 8, 1, add=-2.1415)
        )
        self.conv_con1_phase = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(-2.1415),
        )

        self.conv4_phase = nn.Sequential(
            *conv_phase(1, 4, (5, 5), 1, 3, 2, add=-2.1415)
        )
        self.conv5_phase = nn.Sequential(
            *conv_phase(4, 8, (5, 5), 1, 2, 1, add=-2.1415)
        )
        self.conv6_phase = nn.Sequential(
            *conv_phase(8, 12, (3, 3), 1, 3, 2, add=-2.1415)
        )
        self.conv7_phase = nn.Sequential(
            *conv_phase(12, 16, (3, 3), 1, 1, 1, add=-2.1415)
        )
        self.conv_con2_phase = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(-2.1415),
        )

        self.conv8_phase = nn.Sequential(
            *conv_phase(1, 4, (3, 3), 1, 1, 1, add=-2.1415)
        )
        self.conv9_phase = nn.Sequential(
            *conv_phase(4, 8, (3, 3), 1, 1, 1, add=-2.1415)
        )
        self.conv10_phase = nn.Sequential(
            *conv_phase(8, 12, (3, 3), 1, 2, 2, add=-2.1415)
        )
        self.conv_con3_phase = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(-2.1415),
        )
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))

    def forward(self, x):
        inp = x.clone()
        phase = x[:, 1, :].unsqueeze(1)

        # First block
        phase = self.conv1_phase(phase)

        phase = self.conv2_phase(phase)

        phase = self.conv3_phase(phase)

        phase = self.conv_con1_phase(phase)

        # Second block
        phase = self.conv4_phase(phase)

        phase = self.conv5_phase(phase)

        phase = self.conv6_phase(phase)

        phase = self.conv7_phase(phase)

        phase = self.conv_con2_phase(phase)

        # Third block
        phase = self.conv8_phase(phase)

        phase = self.conv9_phase(phase)

        phase = self.conv10_phase(phase)

        phase = self.conv_con3_phase(phase)

        inp_phase = inp[:, 1].unsqueeze(1)

        x1 = self.symmetry_imag(phase).reshape(-1, 1, phase.shape[2], phase.shape[2])
        x1[inp_phase != 0] = inp_phase[inp_phase != 0]
        return x1


class filter_deep_amp_unc(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = block_1_a()
        self.block2 = block_2_a()
        self.block3 = block_3_a()
        self.block1_unc = block_1_a_unc()
        self.block2_unc = block_2_a_unc()
        self.block3_unc = block_3_a_unc()
        self.symmetry = Lambda(symmetry)
        self.elu = GeneralELU(add=+1)
        self.elu_unc = GeneralELU(add=+1 + 1e-5)

    def forward(self, x):
        x = x[:, 0].unsqueeze(1)
        inp = x.clone()

        # Blocks to predict amp
        x0 = self.block1(x)
        x0 = self.block2(x0)
        x0 = self.block3(x0)

        # x0 = x0.clone()
        # x0 = x0 + inp
        x0 = self.symmetry(x0[:, 0]).reshape(-1, 1, 63, 63)
        # x0 = self.elu(x0)
        x0[inp != 0] = inp[inp != 0]

        # Blocks to predict uncertainty
        x1 = self.block1_unc(x)
        x1 = self.block2_unc(x1)
        x1 = self.block3_unc(x1)

        # x1 = x1.clone()
        x1 = self.symmetry(x1[:, 0]).reshape(-1, 1, 63, 63)
        # x1 = self.elu_unc(x1)
        x1[inp != 0] = 1e-8

        out = torch.cat([x0, x1, inp], dim=1)
        return out


class filter_deep_phase_unc(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = block_1_p()
        self.block2 = block_2_p()
        self.block3 = block_3_p()
        self.block4 = block_4_p()
        self.bridge = bridge()
        self.block1_unc = block_1_p_unc()
        self.block2_unc = block_2_p_unc()
        self.block3_unc = block_3_p_unc()
        self.symmetry = Lambda(partial(symmetry, mode="imag"))
        self.symmetry_unc = Lambda(symmetry)
        self.elu_phase = GeneralELU(add=-(pi - 1), maxv=pi)
        self.elu = GeneralELU(add=+(1e-5))
        self.phase_range = Lambda(phase_range)

    def forward(self, x):
        x = x[:, 1].unsqueeze(1)
        inp = x.clone()

        # Blocks to predict phase
        x0 = self.block1(x)
        x0 = self.block2(x0)
        x0 = self.block3(x0)

        # x0 = x0.clone()
        # x0 = x0 + inp
        # x0 = self.phase_range(x0)
        x0 = self.symmetry(x0[:, 0]).reshape(-1, 1, 63, 63)
        x0[inp != 0] = inp[inp != 0]
        # x0 = self.elu_phase(x0)

        # Blocks to predict uncertainty
        x1 = self.block1_unc(x)
        x1 = self.block2_unc(x1)
        x1 = self.block3_unc(x1)

        x1 = self.symmetry_unc(x1[:, 0]).reshape(-1, 1, 63, 63)
        x1[inp != 0] = 1e-8

        out = torch.cat([x0, x1, inp], dim=1)
        return out


class filter_deep(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.conv1_amp = nn.Sequential(*conv_amp(1, 4, (23, 23), 1, 11, 1))
        self.conv1_phase = nn.Sequential(
            *conv_phase(1, 4, (23, 23), 1, 11, 1, add=1 - pi)
        )
        self.conv2_amp = nn.Sequential(*conv_amp(4, 8, (21, 21), 1, 10, 1))
        self.conv2_phase = nn.Sequential(
            *conv_phase(4, 8, (21, 21), 1, 10, 1, add=1 - pi)
        )
        self.conv3_amp = nn.Sequential(*conv_amp(8, 12, (17, 17), 1, 8, 1))
        self.conv3_phase = nn.Sequential(
            *conv_phase(8, 12, (17, 17), 1, 8, 1, add=1 - pi)
        )
        self.conv_con1_amp = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con1_phase = nn.Sequential(
            LocallyConnected2d(12, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1 - pi),
        )

        self.conv4_amp = nn.Sequential(*conv_amp(1, 4, (5, 5), 1, 4, 2))
        self.conv4_phase = nn.Sequential(*conv_phase(1, 4, (5, 5), 1, 4, 2, add=1 - pi))
        self.conv5_amp = nn.Sequential(*conv_amp(4, 8, (5, 5), 1, 2, 1))
        self.conv5_phase = nn.Sequential(*conv_phase(4, 8, (5, 5), 1, 2, 1, add=1 - pi))
        self.conv6_amp = nn.Sequential(*conv_amp(8, 12, (3, 3), 1, 2, 2))
        self.conv6_phase = nn.Sequential(
            *conv_phase(8, 12, (3, 3), 1, 2, 2, add=1 - pi)
        )
        self.conv7_amp = nn.Sequential(*conv_amp(12, 16, (3, 3), 1, 1, 1))
        self.conv7_phase = nn.Sequential(
            *conv_phase(12, 16, (3, 3), 1, 1, 1, add=1 - pi)
        )
        self.conv_con2_amp = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con2_phase = nn.Sequential(
            LocallyConnected2d(16, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1 - pi),
        )

        self.conv8_amp = nn.Sequential(*conv_amp(1, 4, (3, 3), 1, 1, 1))
        self.conv8_phase = nn.Sequential(*conv_phase(1, 4, (3, 3), 1, 1, 1, add=1 - pi))
        self.conv9_amp = nn.Sequential(*conv_amp(4, 8, (3, 3), 1, 1, 1))
        self.conv9_phase = nn.Sequential(*conv_phase(4, 8, (3, 3), 1, 1, 1, add=1 - pi))
        self.conv10_amp = nn.Sequential(*conv_amp(8, 12, (3, 3), 1, 2, 2))
        self.conv10_phase = nn.Sequential(
            *conv_phase(8, 12, (3, 3), 1, 2, 2, add=1 - pi)
        )
        self.conv11_amp = nn.Sequential(*conv_amp(12, 20, (3, 3), 1, 1, 1))
        self.conv11_phase = nn.Sequential(
            *conv_phase(12, 20, (3, 3), 1, 1, 1, add=1 - pi)
        )
        self.conv_con3_amp = nn.Sequential(
            LocallyConnected2d(20, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.conv_con3_phase = nn.Sequential(
            LocallyConnected2d(20, 1, img_size, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            GeneralELU(1 - pi),
        )
        self.symmetry_real = Lambda(symmetry)
        self.symmetry_imag = Lambda(partial(symmetry, mode="imag"))

    def forward(self, x):
        inp = x.clone()
        amp = x[:, 0, :].unsqueeze(1)
        phase = x[:, 1, :].unsqueeze(1)

        # First block
        amp = self.conv1_amp(amp)
        phase = self.conv1_phase(phase)

        amp = self.conv2_amp(amp)
        phase = self.conv2_phase(phase)

        amp = self.conv3_amp(amp)
        phase = self.conv3_phase(phase)

        amp = self.conv_con1_amp(amp)
        phase = self.conv_con1_phase(phase)

        # Second block
        amp = self.conv4_amp(amp)
        phase = self.conv4_phase(phase)

        amp = self.conv5_amp(amp)
        phase = self.conv5_phase(phase)

        amp = self.conv6_amp(amp)
        phase = self.conv6_phase(phase)

        amp = self.conv7_amp(amp)
        phase = self.conv7_phase(phase)

        amp = self.conv_con2_amp(amp)
        phase = self.conv_con2_phase(phase)

        # Third block
        amp = self.conv8_amp(amp)
        phase = self.conv8_phase(phase)

        amp = self.conv9_amp(amp)
        phase = self.conv9_phase(phase)

        amp = self.conv10_amp(amp)
        phase = self.conv10_phase(phase)

        amp = self.conv11_amp(amp)
        phase = self.conv11_phase(phase)

        amp = self.conv_con3_amp(amp)
        phase = self.conv_con3_phase(phase)

        # amp = amp + inp[:, 0].unsqueeze(1)
        inp_amp = inp[:, 0].unsqueeze(1)
        inp_phase = inp[:, 1].unsqueeze(1)
        # phase = phase + inp[:, 1].unsqueeze(1)
        x0 = self.symmetry_real(amp).reshape(-1, 1, amp.shape[2], amp.shape[2])
        x0[inp_amp != 0] = inp_amp[inp_amp != 0]

        x1 = self.symmetry_imag(phase).reshape(-1, 1, phase.shape[2], phase.shape[2])
        x1[inp_phase != 0] = inp_phase[inp_phase != 0]
        comb = torch.cat([x0, x1], dim=1)
        return comb
