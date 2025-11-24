"""
    加载模型的两种方式：模型+参数。仅参数
    仅参数需要先创建模型架构
    自定义的模型需要将类导入进来才能加载【简单地：from save_model import *】
"""

# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from torch import nn
from save_model import *

# 方式1，加载模型
model = torch.load("vgg16_method1.pth", weights_only=False)
# print(model)

# 方式2，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)

# 陷阱1
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

model = torch.load('tudui_method1.pth', weights_only=False)
print(model)