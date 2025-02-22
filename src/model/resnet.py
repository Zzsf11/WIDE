import torch
import torchvision
import torch.nn as nn
from torchvision.models import resnet 

from model.backbone_base import Backbone
from torchvision.models.resnet import Bottleneck, BasicBlock
from typing import Type, Any, Callable, Union, List, Optional


class ResNet(Backbone):
    def __init__(self, name):
        assert name in ['resnet18', 'resnet50']
        self.name = name
        super(ResNet, self).__init__(get_layers(name))


def get_layers(name):
    if 'resnet18' == name:
        replace_stride_with_dilation=[False, False, False]
        model = resnet.__dict__[name](
                    pretrained=True,
                    replace_stride_with_dilation=replace_stride_with_dilation)
    elif 'resnet50' == name:
        # replace_stride_with_dilation=[False, True, True]
        replace_stride_with_dilation=[False, False, False]
        model = resnet.__dict__[name](
                    pretrained=True,
                    replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        raise ValueError(name)
    
    layer0 = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    layer1 = model.layer1
    layer2 = model.layer2
    layer3 = model.layer3
    layer4 = model.layer4
    return [layer0, layer1, layer2, layer3, layer4]

class My_ResNet(resnet.ResNet):
    def __init__(self, name, **kwargs):
        if name == "resnet50":
            replace_stride_with_dilation=[False, True, True]# 显著增加显存占用
            super(My_ResNet, self).__init__(Bottleneck, [3, 4, 6, 3], replace_stride_with_dilation=replace_stride_with_dilation, **kwargs)
        else:
            super(My_ResNet, self).__init__(BasicBlock, [2, 2, 2, 2], **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return (x1, x2, x3, x4)