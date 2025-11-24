"""
    1、优化器：更新参数，w = w - lr * grad
    optim = torch.optim.SGD(tudui.parameters(), lr=0.01) # 参数、学习率
    optim.zero_grad() # 梯度清零
    optim.step() # 更新操作

    2、模型训练过程【反向传播默认会累加梯度，所以一批数据】
    数据x => 模型model => 预测值pred => 求损失loss(pred, target) => 梯度清零 => 反向传播计算梯度 => 更新参数 ...... 循环每个样本，循环多轮
    考虑 batch
    数据x => 模型model => 预测值pred => 求损失loss(pred, target) => 梯度清零 => 反向传播计算并累加梯度 => 更新参数 ...... 循环每个样本，循环多轮

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
# 导入具体的神经网络层：顺序容器、卷积层、池化层、展平层、线性层
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
# 导入学习率调度器（虽然导入了但未使用）
from torch.optim.lr_scheduler import StepLR
# 导入数据加载器
from torch.utils.data import DataLoader

# 加载CIFAR-10测试数据集
dataset = torchvision.datasets.CIFAR10(
    "./dataset",  # 数据集存储路径
    train=False,  # 使用测试集（10000张图像）
    transform=torchvision.transforms.ToTensor(),  # 将图像转换为张量并归一化到[0,1]
    download=True  # 如果数据集不存在则自动下载
)

# 创建数据加载器，用于批量加载数据
# 注意：batch_size=1，每个批次只有1个样本，训练速度会很慢
dataloader = DataLoader(dataset, batch_size=1)


# 定义自定义神经网络模型Tudui
class Tudui(nn.Module):
    """
    CNN分类网络模型
    功能：构建一个完整的卷积神经网络用于CIFAR-10图像分类
    """

    def __init__(self):
        """
        初始化模型，使用Sequential顺序容器定义网络结构
        """
        # 调用父类nn.Module的初始化方法
        super(Tudui, self).__init__()
        # 使用Sequential顺序容器定义网络层，按顺序执行
        self.model1 = Sequential(
            # 第一层：卷积 + 池化
            Conv2d(3, 32, 5, padding=2),  # 输入通道3(RGB)，输出32，卷积核5x5，填充2保持尺寸
            MaxPool2d(2),  # 2x2最大池化，下采样2倍

            # 第二层：卷积 + 池化
            Conv2d(32, 32, 5, padding=2),  # 输入32，输出32，卷积核5x5，填充2
            MaxPool2d(2),  # 2x2最大池化，下采样2倍

            # 第三层：卷积 + 池化
            Conv2d(32, 64, 5, padding=2),  # 输入32，输出64，卷积核5x5，填充2
            MaxPool2d(2),  # 2x2最大池化，下采样2倍

            # 分类层：展平 + 全连接
            Flatten(),  # 将特征图展平为一维向量
            Linear(1024, 64),  # 全连接层，1024输入特征，64输出特征
            Linear(64, 10)  # 输出层，64输入特征，10输出（对应CIFAR-10的10个类别）
        )

    def forward(self, x):
        """
        前向传播过程
        x: 输入张量，形状为 [batch_size, 3, 32, 32]
        返回: 分类得分，形状为 [batch_size, 10]
        """
        # 将输入通过定义好的顺序模型
        x = self.model1(x)
        return x


# 定义损失函数 - 交叉熵损失，适用于多分类问题
loss = nn.CrossEntropyLoss()
# 创建模型实例
tudui = Tudui()
# 定义优化器 - 随机梯度下降，学习率0.01
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)

# 训练循环：20个epoch
for epoch in range(20):
    running_loss = 0.0  # 初始化当前epoch的累计损失

    # 遍历数据加载器中的所有批次
    for data in dataloader:
        # 解包批次数据
        imgs, targets = data
        # 前向传播：将图像输入模型得到预测输出
        outputs = tudui(imgs)
        # 计算损失：比较预测输出和真实标签
        result_loss = loss(outputs, targets)
        # 反向传播准备：清零梯度（防止梯度累加）
        optim.zero_grad()
        # 反向传播：计算损失相对于模型参数的梯度
        result_loss.backward()
        # 参数更新：根据梯度更新模型参数
        optim.step()
        # 累计当前批次的损失
        running_loss = running_loss + result_loss

    # 打印当前epoch的总损失
    print(running_loss)


### nan,不妨减小lr、batch
# D:\Anaconda3\envs\study\python.exe C:\Users\dy\Desktop\小土堆\实战\optimizer.py
# tensor(18746.1172, grad_fn=<AddBackward0>)
# tensor(16176.6270, grad_fn=<AddBackward0>)
# tensor(15429.7656, grad_fn=<AddBackward0>)
# tensor(15898.4688, grad_fn=<AddBackward0>)
# tensor(17447.1934, grad_fn=<AddBackward0>)
# tensor(20178.0723, grad_fn=<AddBackward0>)
# tensor(22422.9590, grad_fn=<AddBackward0>)
# tensor(24016.4863, grad_fn=<AddBackward0>)
# tensor(24387.7949, grad_fn=<AddBackward0>)
# tensor(25872.6191, grad_fn=<AddBackward0>)
# tensor(25907.2051, grad_fn=<AddBackward0>)
# tensor(26921.4941, grad_fn=<AddBackward0>)
# tensor(27253.4863, grad_fn=<AddBackward0>)
# tensor(28706.1934, grad_fn=<AddBackward0>)
# tensor(30713.5039, grad_fn=<AddBackward0>)
# tensor(34206.9414, grad_fn=<AddBackward0>)
# tensor(nan, grad_fn=<AddBackward0>)
# tensor(nan, grad_fn=<AddBackward0>)
# tensor(nan, grad_fn=<AddBackward0>)
# tensor(nan, grad_fn=<AddBackward0>)
#
# 进程已结束，退出代码为 0