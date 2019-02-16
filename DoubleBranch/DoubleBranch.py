import torch
from torchvision import models
from torch import nn

# 双分支vgg
class DoubleBranch(nn.Module):
    def __init__(self,imgSize=224):
        super(DoubleBranch,self).__init__()
        self.firstBranch = models.vgg16_bn().features
        self.secondBranch = models.vgg16_bn().features
        self.secondBranch[0] = nn.Conv2d(2,64,3,1,1)
        self.fc = models.vgg16_bn().classifier
        fcImgSize = (imgSize//32) * (imgSize//32)
        self.fc[0] = nn.Linear(fcImgSize * 512 * 2,4096)

    def forward(self, x1,x2):
        x1 = self.firstBranch(x1)
        x2 = self.secondBranch(x2)
        x = torch.cat([x1,x2], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net = DoubleBranch()
    print(net)