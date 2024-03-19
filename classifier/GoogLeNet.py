import torch
import torch.nn as nn

'''
param {int} in_c       输入通道数
param {int}   c1       分给路线 1 通道数
param {tuple} c2       分给路线 2 通道数、首层输出通道数
param {tuple} c3       分给路线 3 通道数、首层输出通道数
param {int}   c4       分给路线 4 通道数
param {nn.Module} Activation
description: 其中 p2、p3 包含两个卷积层，因而不仅需要指定分配给其的通道数，
             而且需要指定首个卷积层的输出通道数
example: 下面例子是创建一个inception block 并且使用随机数作为输入测试其能否跑通，
    >>> inception = Inception(192, 64, (96, 128), (16, 32), 32)
    >>> r = torch.randn(3, 192, 224, 224)
    >>> print(inception(r).shape) 
'''  
class Inception(nn.Module):
    def __init__(self, in_c: int, c1: int, c2:tuple, c3:tuple, c4:int, Activation=nn.ReLU):
        super(Inception, self).__init__()
        
        self.p1 = self._build_p1(in_c, c1, Activation)
        self.p2 = self._build_p2(in_c, c2, Activation)
        self.p3 = self._build_p3(in_c, c3, Activation)
        self.p4 = self._build_p4(in_c, c4, Activation)

    # 由于 Inception包含四条路线，并且每条路线未来都有可能拓展，故对每个路线的构造做一个封装
    def _build_p1(self, in_c, c1, Activation):
        return nn.Sequential(nn.Conv2d(in_c, c1, kernel_size=1),
                             nn.BatchNorm2d(c1),
                             Activation())
    def _build_p2(self, in_c, c2, Activation):
        return nn.Sequential(nn.Conv2d(in_c, c2[0], kernel_size=1),
                             nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
                             nn.BatchNorm2d(c2[1]),
                             Activation())
    def _build_p3(self, in_c, c3, Activation):
        return nn.Sequential(nn.Conv2d(in_c, c3[0], kernel_size=1), 
                             nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
                             nn.BatchNorm2d(c3[1]),
                             Activation())
    def _build_p4(self, in_c, c4, Activation):
        return nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
                             nn.Conv2d(in_c, c4, kernel_size=1),
                             nn.BatchNorm2d(c4),
                             Activation())

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return torch.cat((p1, p2, p3, p4), dim=1)


# 只需要重写其中一条路线即可，其它模块均可复用
class InceptionV2(Inception):
    def _build_p3(self, in_c, c3, Activation):
        return nn.Sequential(nn.Conv2d(in_c,  c3[0], kernel_size=1), 
                             nn.Conv2d(c3[0], c3[1], kernel_size=3, padding=1),
                             nn.Conv2d(c3[1], c3[1], kernel_size=3, padding=1),
                             nn.BatchNorm2d(c3[1]),
                             Activation())

# 重写了其中两条路线，使用了矩阵空间分解，把一个 n x n 卷积变成了 1 x n 和 n x 1 卷积
class InceptionV3(Inception):
    def _build_p2(self, in_c, c2, Activation):
        return nn.Sequential(nn.Conv2d(in_c, c2[0], kernel_size=1),
                            nn.Conv2d(c2[0], c2[1], kernel_size = (1,3), padding = (0,1)),
                            nn.Conv2d(c2[1], c2[1], kernel_size = (3,1), padding = (1,0)),
                            nn.BatchNorm2d(c2[1]),
                            Activation())
    def _build_p3(self, in_c, c3, Activation):
        return nn.Sequential(nn.Conv2d(in_c,  c3[0], kernel_size=1),
                            nn.Conv2d(c3[0], c3[0], kernel_size=(1, 3), padding=(0, 1)),
                            nn.Conv2d(c3[0], c3[0], kernel_size=(3, 1), padding=(1, 0)),
                            nn.Conv2d(c3[0], c3[1], kernel_size=(1, 3), padding=(0, 1)),
                            nn.Conv2d(c3[1], c3[1], kernel_size=(3, 1), padding=(1, 0)),
                            nn.BatchNorm2d(c3[1]),
                            Activation())

'''
Analysis: 因为 Inception 模块收集跨尺度特征可能并非其性能提高的主要原因，而是因为模块存在
          近似的 Identity Shortcut 结构，故其能够搭建很深仍然能够有效训练，也正因为这样，
          Inception-ResNet之中添加一条 shortcut 并没有带来明显的涨点。
Reference: https://www.zhihu.com/question/66396783/answer/761129942
'''
class GoogLeNet(nn.Module):
    def __init__(self, input_size: tuple = (None,3,224,224), 
                       num_classes:int = 10, 
                       Activation: nn.Module = nn.ReLU):
        super(GoogLeNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels = input_size[1], out_channels = 64, kernel_size = 7, stride = 2, padding = 3),
            Activation(), 
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        
        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size = 1),
                                Activation(),
                                nn.Conv2d(64, 192, kernel_size = 3, padding = 1),
                                Activation(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding = 1))
        
        self.b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                                Inception(256, 128, (128, 192), (32, 96), 64),
                                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        
        self.b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                                Inception(512, 160, (112, 224), (24, 64), 64), 
                                Inception(512, 128, (128, 256), (24, 64), 64),
                                Inception(512, 112, (114, 288), (32, 64), 64),
                                Inception(528, 256, (160, 320), (32, 128), 128),
                                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        
        self.b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                                Inception(832, 384, (192, 384), (48, 128), 128), 
                                nn.AdaptiveAvgPool2d((1,1)),
                                nn.Flatten())
        self.features = [self.b1, self.b2, self.b3, self.b4, self.b5]
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        for blk in self.features:
            x = blk(x)
        return self.linear(x)


         

if __name__ == '__main__':
    from ops import count_parameters
    r = torch.randn(3, 192, 224, 224)
    
    v2 = InceptionV2(192, 64, (96, 128), (16, 32), 32)
    print(f"Parameters = {count_parameters(v2)}\t", "Output shape = ", v2(r).shape)
    
    v3 = InceptionV3(192, 64, (96, 128), (16, 32), 32)
    print(f"Parameters = {count_parameters(v3)}\t", "Output shape = ", v3(r).shape)
    
    r = torch.randn(3, 3, 224, 224)
    g1 = GoogLeNet()
    print(f"Parameters = {count_parameters(g1)}\t", "Output shape = ", g1(r).shape)
    
    
    
    