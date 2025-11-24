"""
    1、构建线性层
    self.linear1 = Linear(196608, 10) # 输入、输出

    2、展平 flatten
    # 将任意形状的张量展平为一维
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # shape: [2, 2, 2]
    output = torch.flatten(input)  # shape: [8], 值: [1, 2, 3, 4, 5, 6, 7, 8]
"""

# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
# 导入PyTorch深度学习框架
import torch
# 导入torchvision，包含计算机视觉相关的数据集和变换
import torchvision
# 从PyTorch导入神经网络模块
from torch import nn
# 从神经网络模块中导入线性层（全连接层）
from torch.nn import Linear
# 导入数据加载器
from torch.utils.data import DataLoader

# 加载CIFAR-10测试数据集
dataset = torchvision.datasets.CIFAR10(
    "../data",  # 数据集存储路径
    train=False,  # 使用测试集（10000张图像）
    transform=torchvision.transforms.ToTensor(),  # 将图像转换为张量并归一化到[0,1]
    download=True  # 如果数据集不存在则自动下载
)

# 创建数据加载器，用于批量加载数据
dataloader = DataLoader(
    dataset,  # 要加载的数据集
    batch_size=64,  # 每个批次包含64个样本
    drop_last=True  # 丢弃最后一个不完整的批次（确保所有批次大小一致）
)


# 定义自定义神经网络模型Tudui
class Tudui(nn.Module):
    """
    线性层演示模型
    功能：展示线性层（全连接层）的作用和维度变换
    """

    def __init__(self):
        """
        初始化模型，定义线性层
        """
        # 调用父类nn.Module的初始化方法
        super(Tudui, self).__init__()
        # 定义线性层（全连接层）
        self.linear1 = Linear(
            196608,  # 输入特征数：196608个特征
            10  # 输出特征数：10个类别（对应CIFAR-10的10个类别）
        )

    def forward(self, input):
        """
        前向传播过程
        input: 输入张量，形状为 [196608] 或 [batch_size, 196608]
        返回: 线性变换后的输出，形状为 [10] 或 [batch_size, 10]

        数据流示例：
        输入: [196608] → 线性变换 → 输出: [10]
        """
        # 将输入通过线性层进行变换
        # 线性变换公式: output = input × weight^T + bias
        output = self.linear1(input)
        return output


# 创建模型实例
tudui = Tudui()

# 遍历数据加载器，处理每个批次的数据
for data in dataloader:
    # 解包批次数据：imgs为图像张量，targets为标签
    imgs, targets = data
    # 打印输入图像的形状
    # 输出: torch.Size([64, 3, 32, 32])
    # 64: 批次大小, 3: 通道数(RGB), 32: 高度, 32: 宽度
    print(imgs.shape)

    # 将图像张量展平为一维向量
    # torch.flatten() 将多维张量展平为一维
    # 计算: 64 * 3 * 32 * 32 = 64 * 3072 = 196608
    output = torch.flatten(imgs)
    # 打印展平后的形状
    # 输出: torch.Size([196608]) - 一个包含196608个元素的一维向量
    print(output.shape)

    # 将展平后的向量输入线性层进行变换
    output = tudui(output)
    # 打印线性层输出形状
    # 输出: torch.Size([10]) - 对应CIFAR-10的10个类别得分
    print(output.shape)