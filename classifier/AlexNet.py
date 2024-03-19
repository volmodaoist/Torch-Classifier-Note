import torch
import torch.nn as nn
import torchvision.models as models


'''
这个特征提取的配置是一个简化版本，用于 ImageNet 数据集的配置详见自带的网络结构
>>>  model = models.alexnet()
>>>  print(model)
'''
def _alex_featues(input_size: tuple = (None, 224, 224), Activation: nn.Module = nn.ReLU):
    return  nn.Sequential(
        # 先上两个大卷积核对图像进行出来
        nn.Conv2d(in_channels = input_size[1], out_channels = 96, kernel_size = 11, stride = 4), 
        nn.BatchNorm2d(96), Activation(), nn.MaxPool2d(kernel_size = 3, stride = 2), 
        nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = 5, padding = 2),
        nn.BatchNorm2d(256), Activation(), nn.MaxPool2d(kernel_size = 3, stride = 2),

        # 连续三个卷积层
        nn.Conv2d(in_channels = 256, out_channels = 384, kernel_size = 3, padding = 1),  nn.BatchNorm2d(384), Activation(inplace = True), 
        nn.Conv2d(in_channels = 384, out_channels = 384, kernel_size = 3, padding = 1),  nn.BatchNorm2d(384), Activation(inplace = True),
        nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = 3, padding = 1),  nn.BatchNorm2d(256), Activation(inplace = True),
        nn.MaxPool2d(kernel_size = 3, stride = 2)
    )

class AlexNet(nn.Module):
    def __init__(self, input_size: tuple = (None, 3, 224, 224), num_classes:int = 10, Activation: nn.Module = nn.ReLU):
        super(AlexNet, self).__init__()
        self.features = _alex_featues(input_size)
        # 下面三个密集连接层非常占用显存，参数量基本来自于下面这个连接层
        
        self.linear = nn.Sequential(
            nn.Linear(6400, 4096), nn.BatchNorm1d(4096), Activation(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.BatchNorm1d(4096), Activation(), nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
   
    def forward(self, x):
        x = self.features(x).flatten(1, -1)
        x = self.linear(x)
        return x
    
    
class SimpAlexNet(nn.Module):
    def __init__(self, input_size: tuple = (None, 3, 224, 224), num_classes:int = 10, Activation: nn.Module = nn.ReLU):
        super(SimpAlexNet, self).__init__()
        self.features = nn.Sequential(
            *_alex_featues(input_size),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.linear = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), Activation(),
            nn.Dropout(0.5),
            nn.Linear(128, 128), nn.BatchNorm1d(128), Activation(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.linear(x)
        return x



# 使用自带的AlexNet 然后来做一些拓展工作
class BuiltinAlexNet(nn.Module):
    def __init__(self, num_classes=1000, weights_path=None):
        super(BuiltinAlexNet, self).__init__()
        self.alexnet = models.alexnet(pretrained=True if weights_path is None else False)
    
        # 先载入权重, 权重有可能并非参数字典, 而是直接整个模型保存下来的pth
        if weights_path is not None:
            try:
                self.alexnet.load_state_dict(torch.load(weights_path))
            except TypeError:
                self.alexnet = torch.load(weights_path, map_location = torch.device('cpu'))
     
        # self.alexnet.classifier = nn.Identity()
        if num_classes != 1000:
            self.alexnet.classifier[6] = nn.Linear(4096, num_classes)


    def forward(self, x):
        return self.alexnet(x)




        

if __name__ == '__main__':
    pass