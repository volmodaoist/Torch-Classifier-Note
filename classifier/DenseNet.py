import torch
import torch.nn as nn

# 此处采用「批归一化+激活+卷积」这种模式，而非「卷积+批归一化+激活」
# 因为大量实验结果表明先做批归一化再做激活再做卷积的效果会更好一些，这种做法也被叫做预激活(Pre-Activation)
def conv_block(in_channels, out_channels, Activation = nn.ReLU):
    return nn.Sequential(nn.BatchNorm2d(in_channels), Activation(),
                         nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))

# 过渡层的作用主要是避免DenseBlock带来的参数量爆炸性的提升
def transition_block(in_channels, out_channels, Activation = nn.ReLU):
    return nn.Sequential(nn.BatchNorm2d(in_channels), Activation(),
                         nn.Conv2d(in_channels, out_channels, kernel_size = 1),
                         nn.AvgPool2d(kernel_size = 2, stride = 2)) 


class DenseBlock(nn.Module):
    def __init__(self, num_convs, in_channels, out_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(conv_block(in_channels + i * out_channels, out_channels))
        self.features = nn.Sequential(*layer)
    
    def forward(self, X):
        for blk in self.features:
            Y = blk(X)
            X = torch.cat((X, Y), dim = 1)
        return X
        
'''
description: 
    1. 关于代码实现，我们解释一下构造函数之中每个部分的含义：
            首先由于每个 DenseBlock 内部每个卷积的输出都将拼接起来，根据上述实现可知尽管各模块输入通道不同，但其输出通道数是一样的，
            因而我们使用一个变量 num_channels 记录当前 DenseBlock 输出通道数，那么当前通道数可以写为:
        
                nums_channels = DenseBlock卷积模块的数量 (num_cons) x 每个卷积模块的输出通道数量 (growth_rate)
                
    2. 关于参数量，我们希望说明一点，尽管 DenseNet 常被诟病内存或是显存消耗过多，但是由于过渡层的存在，其参数量其实小于一般的 ResNet，
       当然 ResNet 也有自己的降低参数的手段， i.e. 引入 Bottleneck 模块
'''
class DenseNet(nn.Module):

    def __init__(self, input_size: tuple = (None,3,224,224), 
                       num_classes:int = 10, 
                       Activation: nn.Module = nn.ReLU,
                       num_convs_in_denseblock = [4, 4, 4, 4])-> None:
        super(DenseNet, self).__init__()
        num_channels, growth_rate =  64, 32
        blks = []
        for i, num_convs in enumerate(num_convs_in_denseblock):
            # 每个稠密块的输入通道是由上一个稠密块的输出通道决定
            blks.append(DenseBlock(num_convs, num_channels, growth_rate))
            num_channels += num_convs * growth_rate
            # 每个稠密块身后添加一个过渡层用于控制通道数量，以免参数过多
            if i != len(num_convs_in_denseblock) - 1:
                 blks.append(transition_block(num_channels, num_channels // 2, Activation))
                 num_channels  = num_channels // 2
        
        self.features = nn.Sequential(nn.Conv2d(input_size[1], 64, kernel_size = 7, stride = 2, padding = 3), 
                                      nn.BatchNorm2d(64), Activation(), 
                                      nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                                      *blks,
                                      nn.BatchNorm2d(num_channels), Activation(), 
                                      nn.AdaptiveAvgPool2d((1,1)),
                                      nn.Flatten())
        self.linear = nn.Linear(num_channels, num_classes)
        
    def forward(self, X):
        X = self.features(X)
        X = self.linear(X)
        return X 
    
if __name__ == '__main__':
    r = torch.randn(3, 3, 224, 224)
    
    dn = DenseNet()
    print(dn(r).shape)
    
    dn_121 = DenseNet(num_convs_in_denseblock = [6, 12, 24, 16])
    print(dn_121(r).shape)
    
    dn_169 = DenseNet(num_convs_in_denseblock = [6, 12, 32, 32])
    print(dn_169(r).shape)
    
    dn_264 = DenseNet(num_convs_in_denseblock = [6, 12, 64, 48])
    print(dn_264(r).shape) 
    