"""
    1、构建神经网络：继承nn.Module，
    (1)调用父类nn.Module的初始化方法，这是必须的。 super().__init__()【定义层】
    (2) 定义前向传播过程，这是每个nn.Module子类必须实现的方法【使用层】
    2、卷积层
    from torch.nn import Conv2d
    self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0 )
    【输入通道、输出通道、卷积核、步长、填充】
    【3=>3×3】
    3、输入是四维的(批次、通道、高、宽)。可以通过reshape处理

"""


# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
# 导入PyTorch深度学习框架
import torch
# 导入torchvision，包含计算机视觉相关的数据集、模型和变换
import torchvision
# 从PyTorch导入神经网络模块
from torch import nn
# 从神经网络模块中导入二维卷积层
from torch.nn import Conv2d
# 导入数据加载器
from torch.utils.data import DataLoader
# 导入TensorBoard可视化工具
from torch.utils.tensorboard import SummaryWriter

# 加载CIFAR-10测试数据集
# CIFAR-10包含10个类别的60000张32x32彩色图像，这里使用测试集
dataset = torchvision.datasets.CIFAR10(
    "./dataset",  # 数据集存储路径
    train=False,  # 使用测试集（10000张图像）
    transform=torchvision.transforms.ToTensor(),  # 将图像转换为PyTorch张量，归一化到[0,1]
    download=True  # 如果数据集不存在则自动下载
)

# 创建数据加载器，用于批量加载数据
dataloader = DataLoader(
    dataset,  # 要加载的数据集
    batch_size=64  # 每个批次包含64个样本
)


# 定义自定义神经网络模型Tudui
class Tudui(nn.Module):
    """
    卷积神经网络演示模型
    功能：展示卷积层的作用和特征图变化
    """

    def __init__(self):
        """
        初始化模型，定义网络层
        """
        # 调用父类nn.Module的初始化方法
        super(Tudui, self).__init__()
        # 定义二维卷积层
        self.conv1 = Conv2d(
            in_channels=3,  # 输入通道数：RGB图像有3个通道
            out_channels=6,  # 输出通道数：生成6个特征图
            kernel_size=3,  # 卷积核大小：3x3
            stride=1,  # 步长：1
            padding=0  # 填充：0
        )

    def forward(self, x):
        """
        前向传播过程
        x: 输入张量，形状为 [batch_size, 3, 32, 32]
        返回: 卷积后的特征图

        数据流示例：
        输入: [64, 3, 32, 32] → 卷积 → 输出: [64, 6, 30, 30]
        """
        # 将输入通过卷积层
        x = self.conv1(x)
        # 返回卷积结果
        return x


# 创建模型实例
tudui = Tudui()

# 创建TensorBoard的SummaryWriter，日志保存在上级目录的logs文件夹中
writer = SummaryWriter("logs_conv")

# 初始化步骤计数器，用于TensorBoard中的全局步数
step = 0

# 遍历数据加载器，处理每个批次的数据
for data in dataloader:
    # 解包批次数据：imgs为图像张量，targets为标签
    imgs, targets = data
    # 将图像输入模型进行前向传播，得到卷积后的输出
    output = tudui(imgs)

    # 打印输入和输出的形状，用于调试和验证
    print(imgs.shape)  # 输入形状：[64, 3, 32, 32]
    print(output.shape)  # 输出形状：[64, 6, 30, 30]

    # 将原始输入图像写入TensorBoard
    # torch.Size([64, 3, 32, 32]) - 64张32x32的RGB图像
    writer.add_images("input", imgs, step)

    # torch.Size([64, 6, 30, 30]) -> 需要重塑为 [xxx, 3, 30, 30] 才能在TensorBoard中显示
    # 因为TensorBoard的add_images要求通道数为1或3，但输出是6个通道

    # 重塑输出张量，使其可以被TensorBoard显示
    # -1: 自动计算该维度大小，保持总元素数不变
    # 3: 将6个通道重组为3个通道（因为6÷3=2，所以批次大小会翻倍）
    # 30, 30: 保持特征图的空间尺寸不变
    output = torch.reshape(output, (-1, 3, 30, 30))

    # 将卷积后的输出图像写入TensorBoard
    writer.add_images("output", output, step)

    # 步骤计数器加1，准备记录下一个批次
    step = step + 1

# 关闭SummaryWriter，确保所有数据写入磁盘
writer.close()