from radionets.dl_framework.architectures.basics import *
from radionets.dl_framework.architectures.unet import *
from radionets.dl_framework.architectures.filter import *
from radionets.dl_framework.architectures.filter_deep import *
from radionets.dl_framework.architectures.source_list import *
from radionets.dl_framework.model import Lambda, reshape, unsqueeze0
from torchvision.transforms.functional import crop

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

class CropUNet(nn.Module):
    def __init__(self, unet, cnn, n):
        super().__init__()

        self.unet = unet
        self.cnn = cnn

        self.n = n

        self.reshape = Lambda(reshape)

        self.thresh = torch.nn.Threshold(0.4999, 0)

        self.pad1 = torch.nn.ZeroPad2d((0, 10, 0, 10))
        self.pad2 = torch.nn.ZeroPad2d(22)

        self.unsqueeze = Lambda(unsqueeze0)

    def forward(self, x):
        out_unet = self.unet(x)
        seg = self.reshape(out_unet)
        seg = self.thresh(seg)
        im, mx, my = torch.where(seg != 0)

        x = self.pad1(x)
        out = torch.zeros(x.shape[0], self.n+3)
        for i in range(len(mx)):
            if mx[i] < 10:
                mx[i]=10
            if my[i] < 10:
                my[i]=10
            inp_cnn = crop(
                x[im[i]], top=mx[i]-10, left=my[i]-10, height=20, width=20
            )
            if im[i-1] != im[i]:
                j = i
            if i-j < self.n+3:
                inp_cnn = self.unsqueeze(self.pad2(inp_cnn))
                out[im[i], i-j] = self.cnn(inp_cnn)
        return out_unet, out
