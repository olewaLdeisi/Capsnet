"""
-------------------------------------------------
   File Name：     Resnet50
   Description :
   Author :        lin
   Software:       PyCharm
   date：          2019/2/3 21:31
-------------------------------------------------
   Change Activity:
                   2019/2/3 21:31
-------------------------------------------------
"""
__author__ = 'lin'
from torch import nn
from torchvision import models

class Resnet(nn.Module):
    def __init__(self,n_classes=2,pretrained=False):
        super(Resnet,self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(2048,n_classes)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        x = self.activation(x)
        return x

if __name__ == '__main__':
    import torch
    net = Resnet()
    inputs = torch.Tensor(1,3,224,224)
    out = net(inputs)
    print(1)