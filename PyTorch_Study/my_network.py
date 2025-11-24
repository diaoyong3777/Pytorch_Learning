"""

    1、使用Sequential容器构建完整的CNN分类网络
    from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
    Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    2、Flatten()会保留批次 [64, 64, 4, 4] => [64, 1024]
    区别于torch.flatten() [64, 64, 4, 4] => [64 × 1024]

    3、在tensorboard中可视化计算图
    writer.add_graph(tudui, input) # 模型、输入

    4、数据流
    输入: [64, 3, 32, 32]
    ↓ Conv2d(3,32,5,padding=2) → [64, 32, 32, 32]  (填充保持尺寸)
    ↓ MaxPool2d(2) → [64, 32, 16, 16]              (下采样2倍)
    ↓ Conv2d(32,32,5,padding=2) → [64, 32, 16, 16] (填充保持尺寸)
    ↓ MaxPool2d(2) → [64, 32, 8, 8]                (下采样2倍)
    ↓ Conv2d(32,64,5,padding=2) → [64, 64, 8, 8]   (填充保持尺寸)
    ↓ MaxPool2d(2) → [64, 64, 4, 4]                (下采样2倍)
    ↓ Flatten() → [64, 1024]                       (64×4×4=1024)
    ↓ Linear(1024,64) → [64, 64]                   (全连接层)
    ↓ Linear(64,10) → [64, 10]                     (输出层)
    输出: [64, 10]

    扩展知识
    - 卷积层效果演示：https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    - 计算padding: https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d有公式
"""

# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
# 导入PyTorch深度学习框架
import torch
# 从PyTorch导入神经网络模块
from torch import nn
# 导入具体的神经网络层：卷积层、池化层、展平层、线性层、顺序容器
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
# 导入TensorBoard可视化工具
from torch.utils.tensorboard import SummaryWriter


# 定义自定义神经网络模型Tudui
class Tudui(nn.Module):
    """
    顺序神经网络模型
    功能：演示使用Sequential容器构建完整的CNN分类网络
    """

    def __init__(self):
        """
        初始化模型，使用Sequential顺序容器定义网络结构
        """
        # 调用父类nn.Module的初始化方法
        super(Tudui, self).__init__()
        # 使用Sequential顺序容器定义网络层，按顺序执行
        self.model1 = Sequential(
            # 第一层：卷积层 + 池化层
            Conv2d(3, 32, 5, padding=2),  # 输入通道3(RGB)，输出通道32，卷积核5x5，填充2
            MaxPool2d(2),  # 2x2最大池化，下采样2倍

            # 第二层：卷积层 + 池化层
            Conv2d(32, 32, 5, padding=2),  # 输入通道32，输出通道32，卷积核5x5，填充2
            MaxPool2d(2),  # 2x2最大池化，下采样2倍

            # 第三层：卷积层 + 池化层
            Conv2d(32, 64, 5, padding=2),  # 输入通道32，输出通道64，卷积核5x5，填充2
            MaxPool2d(2),  # 2x2最大池化，下采样2倍

            # 分类层：展平 + 全连接层
            Flatten(),  # 将多维特征图展平为一维向量
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


# 创建模型实例
tudui = Tudui()
# 打印模型结构，查看所有层的信息
print(tudui)

# 创建测试输入数据：64张32x32的RGB图像，所有像素值为1（白色图像）
# 形状: [64, 3, 32, 32] - 批次大小64，3通道，32x32分辨率
input = torch.ones((64, 3, 32, 32))
# 将输入数据通过模型进行前向传播，得到输出
output = tudui(input)
# 打印输出形状，验证网络计算是否正确
print(output.shape)

# 创建TensorBoard的SummaryWriter，日志保存在"../logs_my_network"文件夹中
writer = SummaryWriter("./logs_my_network")
# 将模型的计算图添加到TensorBoard中，便于可视化网络结构
# 参数：模型实例，输入数据（用于追踪计算图）
writer.add_graph(tudui, input)
# 关闭SummaryWriter，确保所有数据写入磁盘
writer.close()