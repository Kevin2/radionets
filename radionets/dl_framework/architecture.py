from radionets.dl_framework.architectures.basics import *
from radionets.dl_framework.architectures.unet import *
from radionets.dl_framework.architectures.filter import *
from radionets.dl_framework.architectures.filter_deep import *
from radionets.dl_framework.architectures.source_list import *
from radionets.dl_framework.model import Lambda, reshape

class MixedUNet(nn.Module):
    def __init__(self, unet, cnn):
        super().__init__()

        self.unet = unet
        self.cnn = cnn

        self.reshape = Lambda(reshape)

    def forward(self, x):
        out_unet = self.unet(x)
        inp_cnn = self.reshape(out_unet)

        inp_cnn = torch.stack([inp_cnn, x], dim=1)
        out = self.cnn(inp_cnn)

        return out_unet, out
