import torch
import torch.nn as nn


'''
param {*} num_convs         vgg_block 包含多少卷积层
param {*} in_channels       vgg_block 输入通道数
param {*} out_channels      vgg_block 输出通道数
param {*} activate          vgg_block 使用何种激活函数
return {nn.Module}          返回一个 VGG 模型的基本模块 VGG Block

description: 这个函数用于构建 VGG 基本模块，一个基本模块包含卷积层与池化层(汇聚层)
             每个基本模块的卷积部分不影响输入特征图的尺寸，尺寸只在经过池化之后 h、w 各自减半
'''
def vgg_block(num_convs, in_channels, out_channels, Activation):
    layers = [] 
    for _ in range(num_convs):
        layers += [nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
                   nn.BatchNorm2d(out_channels), 
                   Activation(inplace = True)]
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size = 2,stride = 2))
    return nn.Sequential(*layers)


'''
原始VGG网络包含五个卷积模块的，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。 
首个卷积层输出通道个数 64，后续每个模块将输出通道数量翻倍，直到 512 停止扩增。
按照下面默认的 conv_arch 构建所得 VGG，由于卷积层个数 8、线性层个数 3，
共有含可学习参数模块个数 11，故称 VGG-11 模型。
'''
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

class VGG(nn.Module):
    def __init__(self, input_size: tuple = (None,3,224,224), num_classes:int = 10, Activation: nn.Module = nn.ReLU,
                 conv_arch = conv_arch):
        super(VGG, self).__init__()
        conv_blks, in_channels = [], input_size[1] 
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels, Activation))
            in_channels = out_channels
        self.features = nn.Sequential(*conv_blks)
        self.gap = nn.AdaptiveMaxPool2d((7, 7))
        self.linear = nn.Sequential(
            nn.Linear(in_channels * 7 * 7, 4096),
            Activation(), nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            Activation(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes) 
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1, -1)
        x = self.linear(x)
        return x

'''
如果直接压平变更向量会导致参数量暴增的模块，为此在学习 NiN 网络之后，
我们借鉴其中的思想对于特征图进行全局平均池化，再丢给全连接层，抛弃最初那种耗费参数的方法！
'''
class SimpVGG(VGG):
    def __init__(self, input_size: tuple = (None,3,224,224), num_classes:int = 10, Activation: nn.Module = nn.ReLU,
                 conv_arch = conv_arch):
        super(SimpVGG, self).__init__(input_size=input_size, num_classes=num_classes, Activation=Activation, conv_arch=conv_arch)
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 
        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            Activation(),
            nn.Linear(256, 128),
            Activation(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1, -1)
        x = self.linear(x)
        return x
    


'''
paper:
    - 文章标题 Very Deep Convolutional Networks for Large-Scale Image Recognition 
    - 原文链接 https://browse.arxiv.org/pdf/1409.1556.pdf
 
description: 文章的核心思想是使用深层的小卷积核代替大卷积核，这样能以更少的参数量达成相同的感受野！
             此外阅读文章会发现 VGG 最深也就十九层，与同年十二月 ResNet 上百层的网络完全不在一个量级！
             文章里面给出了 11/11/13/16/16/19 weighted layer 几种方案，
             此时我们比较叛逆的实现 vgg-8/11/16 三种方案
'''
if __name__ == '__main__':
    from ops import count_parameters
    
    r = torch.randn(3, 3, 224, 224)
    
    vgg8 = VGG(conv_arch = ((1, 64), (1, 128), (1, 256), (2, 512)))
    print(f"Parameters = {count_parameters(vgg8)}\t", "Output shape = ", vgg8(r).shape)

    vgg11 = VGG(conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))
    print(f"Parameters = {count_parameters(vgg11)}\t", "Output shape = ", vgg11(r).shape)
    
    vgg16 = VGG(conv_arch = ((2, 64), (2, 128), (2, 256), (3, 512), (3, 512)))
    print(f"Parameters = {count_parameters(vgg16)}\t", "Output shape = ", vgg16(r).shape)

    # 下面几个网络是在结合了NiN网络思想之后设计的
    # 最主要的改变是将特征提取得到的所有特征图取其全局最大池化，将此向量用于线性层
    
    nin_vgg8 = SimpVGG(conv_arch = ((1, 64), (1, 128), (1, 256), (2, 512)))
    print(f"Parameters = {count_parameters(nin_vgg8)}\t", "Output shape = ", nin_vgg8(r).shape)
        
    nin_vgg11 = SimpVGG(conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)))
    print(f"Parameters = {count_parameters(nin_vgg11)}\t", "Output shape = ", nin_vgg11(r).shape)

    nin_vgg16 = SimpVGG(conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)))
    print(f"Parameters = {count_parameters(nin_vgg16)}\t", "Output shape = ", nin_vgg16(r).shape)    
    
    vgg_blackbone = nn.Sequential(nin_vgg16.features)
    
    print(vgg_blackbone(r).flatten(1, -1).shape)
    

 