import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.up = nn.ConvTranspose2d(in_ch , in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1 -- > x5 40 * 40 * 512
        # x2 -- > x4 80 * 80 * 256
        x1 = self.up(x1)  # x1 80 * 80 * 256

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 默认常数填充，填充0，边缘填充
        x1 =  F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1) # 80 * 80 * 512
        x = self.conv(x) # 80 * 80 * 128
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        # down4
        self.maxd2f = nn.MaxPool2d(2,2)
        self.final1 = nn.Sequential(nn.Conv2d(256,512,3,1,1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True))
        self.avg = nn.AvgPool2d(7)
        self.final2 = nn.Sequential(nn.Conv2d(512,512,3,1,1),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU(True))
        # self.down4 = down(256, 256)
        self.up1 = up(512, 256, bilinear=False)
        self.up2 = up(256, 128, bilinear=False)
        self.up3 = up(128, 64, bilinear=False)
        self.up4 = up(64, 32, bilinear=False)
        self.outc = outconv(32, 1)
        self.classfier = nn.Sequential(nn.Linear(5*5*512,2048),
                                       nn.Linear(2048,2))

    def forward(self, x): # x 640 * 640 * 3
        x1 = self.inc(x) # x1 640 * 640 * 32
        x2 = self.down1(x1) # x2 320 * 320 * 64
        x3 = self.down2(x2) # x3 160 * 160 * 128
        x4 = self.down3(x3) # x4 80 * 80 * 256
        temp = self.final1(self.maxd2f(x4))
        x5 = self.final2(temp)
        # x5 = self.down4(x4) # x5 40 * 40 * 512
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        temp = self.avg(temp).view(x.size(0), -1)
        temp = self.classfier(temp)
        temp = nn.Softmax(dim=1)(temp)
        return temp,nn.Sigmoid()(x)

if __name__ == '__main__':
    net = UNet(3, 1)
    net.cuda()
    x = torch.Tensor(1,3,640,640)
    x = x.cuda()
    out,_ = net(x)
    print(net)
    print(out)

    from torchvision.models import resnet50
    net = resnet50()
    print(net)