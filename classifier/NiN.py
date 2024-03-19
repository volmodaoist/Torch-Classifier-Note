import torch
import torch.nn as nn

def nin_block(in_channels, out_channels, kernel_size, stride, padding, Activation = nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_channels =  in_channels,  out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
        Activation(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 1),
        Activation(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 1),
        Activation()
    )

# 使用两个的 1x1 卷积神经网络等价的实现按像素执行的全卷积神经网络
class NiN(nn.Module):
    def __init__(self, input_size: tuple = (None,3,224,224), 
                       num_classes:int = 10, 
                       Activation: nn.Module = nn.ReLU): 
        super(NiN, self).__init__()
        _, input_channels, _, _ = input_size
        self.features = nn.Sequential(
           nin_block(input_channels, 96, kernel_size = 11, stride = 4, padding = 0, Activation = Activation), 
           nn.MaxPool2d(3, stride = 2),
           nin_block(96, 256, kernel_size = 5, stride = 1, padding = 2, Activation = Activation),
           nn.MaxPool2d(3, stride = 2),
           nin_block(256, 384, kernel_size = 3, stride = 1, padding = 1, Activation = Activation),
           nn.Dropout(0.5),
           nin_block(384, num_classes, kernel_size = 3, stride = 1, padding = 1, Activation = Activation),
           nn.AdaptiveAvgPool2d((1, 1)),
           nn.Flatten()
        )
    
    
    def forward(self, x):
        return self.features(x)


if __name__ == '__main__':
    from ops import count_parameters
    r = torch.randn(3, 3, 224, 224)
    
    nin = NiN()
    print(f"Parameters = {count_parameters(nin)}\t", "Output shape = ", nin(r).shape)