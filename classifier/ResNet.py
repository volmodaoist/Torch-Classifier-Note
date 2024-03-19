import torch
import torch.nn as nn
import torch.nn.functional as F

'''
param {int}  in_channels       输入通道数
param {int}  out_channels      输出通道数
param {bool} use_1x1conv       是否使用 1x1 卷积核调整通道数
param {int}  strides           卷积核的步长
description: 构建一个 ResNet 网络的基本模块, 其中恒等映射连接的实现要求输入输出
             具有相同的通道数，这是一个比较严格的要求，因而增加 use_1x1conv 参数
             允许用户调整输入通道数
example: 
    >>> r = torch.randn(3, 3, 224, 224)
    >>> block = ResBlock(3, 10, use_1x1conv = True)
    >>> print(block(r).shape)
'''   

class ResBlock(nn.Module):  
 
    def __init__(self, in_channels: int, out_channels: int, strides: int = 1, use_1x1conv: bool = False, 
                 Activation = nn.ReLU):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                               kernel_size = 3,  padding = 1, stride = strides)
        self.conv2 = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, 
                               kernel_size = 3, padding = 1)
        self.shortcut = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                               kernel_size = 1, stride = strides) if use_1x1conv else nn.Identity()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self._convs = (self.conv1, self.conv2)
        self._bns = (self.bn1, self.bn2)
        self._Activation = Activation
        
    def forward(self, X):
        A, activate = X, self._Activation()
        for conv, bn in zip(self._convs, self._bns):
            Y = bn(conv(A))
            A = activate(Y)
        Y += self.shortcut(X)
        return Y



def resnet_block(in_channels: int, out_channels: int, num_blocks: int, 
                 Activation = nn.ReLU, first_block = False):
    blk = []
    for i in range(num_blocks):
        if i == 0 and not first_block:
            blk.append(ResBlock(in_channels, out_channels, strides = 2, 
                                    use_1x1conv = True, Activation = Activation))
        else:
            blk.append(ResBlock(out_channels, out_channels, strides = 1, 
                                    use_1x1conv = False, Activation = Activation))
    return blk


class ResNet(nn.Module):
    def __init__(self, input_size: tuple = (None,3,224,224), num_classes:int = 10, Activation = nn.ReLU):
        super(ResNet, self).__init__()
        b1 = nn.Sequential(nn.Conv2d(input_size[1], 64, kernel_size = 7, stride = 2, padding = 3),
                  nn.BatchNorm2d(64), nn.ReLU(),
                  nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, Activation, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2, Activation))
        b4 = nn.Sequential(*resnet_block(128, 256, 2, Activation))
        b5 = nn.Sequential(*resnet_block(256, 512, 2, Activation))
        
        self.features = nn.Sequential(b1, b2, b3, b4, b5,
                                    nn.AdaptiveAvgPool2d((1,1)),
                                    nn.Flatten())
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    model = ResNet()
    print(model(torch.randn(3, 3, 224, 224)).shape) 
 
        
        
        
    
        