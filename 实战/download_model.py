"""
    1、知道模型下载路径
    2、会加载模型、修改模型
"""

# 可以指定路径。os.environ['TORCH_HOME'] = r'C:\Users\Administrator\.cache\torch\hub\checkpoints'【当前进程有效】
# 系统环境变量（真正永久）
# Windows: 系统属性 → 高级 → 环境变量
# 新建系统变量：
# 变量名: TORCH_HOME
# 变量值: C:\Users\Administrator\.cache\torch\hub\checkpoints


# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torchvision
from torch import nn

# 加载模型
vgg16_true = torchvision.models.vgg16(pretrained=True)
vgg16_false = torchvision.models.vgg16(pretrained=False)
# Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to C:\Users\dy/.cache\torch\hub\checkpoints\vgg16-397923af.pth
# 100%|██████████| 528M/528M [00:36<00:00, 15.2MB/s]

# 添加层
print(vgg16_true)
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)
# (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=1000, bias=True)
#     (add_linear): Linear(in_features=1000, out_features=10, bias=True)
#   )
#   (add_linear): Linear(in_features=1000, out_features=10, bias=True)
# )

# 修改层
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
# (classifier): Sequential(
#     (0): Linear(in_features=25088, out_features=4096, bias=True)
#     (1): ReLU(inplace=True)
#     (2): Dropout(p=0.5, inplace=False)
#     (3): Linear(in_features=4096, out_features=4096, bias=True)
#     (4): ReLU(inplace=True)
#     (5): Dropout(p=0.5, inplace=False)
#     (6): Linear(in_features=4096, out_features=10, bias=True)
#   )
# )