import torch
import torch.nn as nn
import torch.optim as optim

from ops import *



'''
description: 主要用于度量两个图像的相似度，而且能够用于多模态学习
             关于孪生神经网络能做的内容其实很多，未来再进一步探索: https://zhuanlan.zhihu.com/p/416227622
example: 
    >>> net = LeNet()
    >>> siam = SiameseNet(input_size = (1, 1, 28, 28), backbone = net)
        
    >>> r1 = torch.randn(1, 1, 28, 28)
    >>> r2 = torch.randn(1, 1, 28, 28)
    >>> print(siam((r1, r2)))
'''
class SiameseNet(nn.Module):
    def __init__(self, input_size: tuple = (None,3,224,224), backbone: nn.Module = None):
        super(SiameseNet, self).__init__()
        assert backbone != None
        self.backbone_features = backbone.features
        
        _, c, h, w = input_size
        _, fshape = self.backbone_features(torch.randn(3, c, h, w)).flatten(1, -1).shape
        
        self.linear = nn.Sequential(
            nn.Linear(fshape, 512),
            nn.Sigmoid(),
            nn.Linear(512, 1),
        )
        
        
    def forward(self, pairs):
        x1, x2 = pairs
        x1 = self.backbone_features(x1).flatten(1, -1)
        x2 = self.backbone_features(x2).flatten(1, -1)
        delta = torch.abs(x1 - x2)
        score = self.linear(delta)
        return score
        


if __name__ == '__main__':
    pass