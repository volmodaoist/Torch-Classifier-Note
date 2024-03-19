import torch
import torch.nn as nn

'''
此处需要说明几个问题：
    1.  模型使用的几个参数与返回值，LeNet 网络相当于 ConvNet 领域的 Hello World，之后我们编写的更加复杂的网络，
        我们编写的网络结构都使用了类似的参数：
            - input_channels  输入图像通道个数
            - num_classes     预测类别个数
            - Activation      激活函数
        至于模型的返回值其实就是 logit 向量 (未经过 softmax 转化的最后一个向量)
    
    2.  早期的 LeNet 使用 Sigmoid 作为激活函数，但是我们改用ReLU，这样精度会更高，如果
        使用古早使其的激活函数，要把输入数据的取值范围变成[-1, 1]，使用 ReLU 则为 [0,1]
    
    3.  forward 过程没有使用 softmax，这是因为 CE 损失函数已经帮我们实现了 log_softmax
        而且如果某个值在向量之中是最大值，经过 softmax 之后仍是最大值，换言之，这个激活
        是保序的，即使不做的 softmax 运算，我们使用 argmax 得到预测类别仍然概率最大类别   
       
    4.  使用 Batch Normalization 意味着 Batch 必须足够大，batch = 1 无法使用这个模块，
        这个模块的意义是令每一层的中间结果的分布都符合均值为零、方差为一，从而加速收敛，至于为什么有效，
        其实目前也没有准确的说法，一些研究将其与贝叶斯先验关联：
            - Bayesian uncertainty estimation for batch normalized deep networks.
            - Towards understanding regularization in batch normalization. 
            
        另外，模型的 train 模式、eval 模式，BN 模块的作用是不一样的，在训练过程中，训练过程之中，
        我们无法得知使用整个数据集来估计平均值和方差，所以是根据每个小批次的平均值和方差训练模型的，
        而在预测模式下，可以根据整个数据集精确计算批量规范化所需的平均值和方差。
'''

class LeNet(nn.Module):
    def __init__(self, input_size: tuple = (None, 1, 28, 28), 
                       num_classes: int = 10, 
                       Activation: nn.Module = nn.Sigmoid):
        super(LeNet, self).__init__()
        _, c, h, w = input_size
        self.features = nn.Sequential(nn.Conv2d(in_channels = c, out_channels = 6, kernel_size = 5, padding = 2),
                                      nn.BatchNorm2d(6),  Activation(), 
                                      nn.MaxPool2d(kernel_size = 2, stride = 2),
                                      nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5, padding = 2), 
                                      nn.BatchNorm2d(16), Activation(),
                                      nn.MaxPool2d(kernel_size = 2, stride = 2),
                                      nn.Flatten())
        h, w = h // 4, w // 4
        self.linear = nn.Sequential(nn.Linear(16 * h * w, 120), nn.BatchNorm1d(120), Activation(), 
                                    nn.Linear(120, 84),  nn.BatchNorm1d(84), Activation(),
                                    nn.Linear(84, num_classes))
    
    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x # logit

    def forward_with_intermediate(self, x):
        intermediates = []
        for module in self.features:
            x = module(x)
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.MaxPool2d):
                intermediates.append(x.view(x.size(0), -1).detach().numpy())

        x = x.view(x.size(0), -1)
        for module in self.linear:
            x = module(x)
            if isinstance(module, nn.Linear):
                intermediates.append(x.detach().numpy())

        return intermediates
        
        
if __name__ == '__main__':
    from ops import count_parameters
    model = LeNet(Activation = nn.ReLU)
    
    r = torch.randn(32, 1, 28, 28) 
    print(f"Parameters = {count_parameters(model)}\t", "Output shape = ", model(r).shape)