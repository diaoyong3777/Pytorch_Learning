"""
    1、构建非线性激活层
    self.relu1 = ReLU()
    self.sigmoid1 = Sigmoid()
    2、没有输入维度要求，输出维度不变

    扩展知识：
    # ReLU: f(x) = max(0, x)
    # 输入: [1, -0.5, -1, 3] → 输出: [1, 0, 0, 3]
    # 特点：简单计算，解决梯度消失，但会"杀死"负值神经元
    # Sigmoid: f(x) = 1 / (1 + e^(-x))
    # 输入: [1, -0.5, -1, 3] → 输出: [0.73, 0.38, 0.27, 0.95]
    # 特点：将值压缩到(0,1)，适合概率输出，但容易梯度消失

    关键理解点：
    ToTensor自动归一化：transforms.ToTensor() 已经将0-255的像素值转换为0-1的浮点数

    激活函数选择：
    ReLU：适合隐藏层，计算简单，缓解梯度消失
    Sigmoid：适合输出层，将值映射到(0,1)表示概率

    实际应用：
    在隐藏层通常使用ReLU
    在二分类输出层使用Sigmoid
    在多分类输出层使用Softmax
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
# 从神经网络模块中导入ReLU和Sigmoid激活函数
from torch.nn import ReLU, Sigmoid
# 导入数据加载器
from torch.utils.data import DataLoader
# 导入TensorBoard可视化工具
from torch.utils.tensorboard import SummaryWriter

# 加载CIFAR-10测试数据集
dataset = torchvision.datasets.CIFAR10(
    "./dataset",  # 数据集存储路径
    train=False,  # 使用测试集（10000张图像）
    download=True,  # 如果数据集不存在则自动下载
    transform=torchvision.transforms.ToTensor()  # 关键：将图像转换为张量并自动归一化到[0,1]
)

# 创建数据加载器，用于批量加载数据
dataloader = DataLoader(
    dataset,  # 要加载的数据集
    batch_size=64  # 每个批次包含64个样本
)


# 定义自定义神经网络模型Tudui
class Tudui(nn.Module):
    """
    激活函数演示模型
    功能：展示Sigmoid激活函数对图像的影响
    """

    def __init__(self):
        """
        初始化模型，定义激活函数层
        """
        # 调用父类nn.Module的初始化方法
        super(Tudui, self).__init__()
        # 定义ReLU激活函数层（虽然定义了但未在forward中使用）
        self.relu1 = ReLU()
        # 定义Sigmoid激活函数层
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        """
        前向传播过程
        input: 输入张量，形状为 [batch_size, 3, 32, 32]，值范围[0,1]
        返回: 经过Sigmoid激活后的特征图，值范围(0,1)

        数据流示例：【RelU不会改变，所以不做展示】
        输入: [0.2, 0.8, 0.5] → Sigmoid → 输出: [0.55, 0.69, 0.62]
        """
        # 将输入通过Sigmoid激活函数
        # Sigmoid公式: σ(x) = 1 / (1 + e^(-x))
        output = self.sigmoid1(input)
        return output


# 创建模型实例
tudui = Tudui()

# 创建TensorBoard的SummaryWriter，日志保存在"../logs_sigmoid"文件夹中
writer = SummaryWriter("./logs_sigmoid")

# 初始化步骤计数器，用于TensorBoard中的全局步数
step = 0

# 遍历数据加载器，处理每个批次的数据
for data in dataloader:
    # 解包批次数据：imgs为图像张量，targets为标签
    imgs, targets = data
    # 将原始输入图像写入TensorBoard
    # 注意：这里的图像值已经在[0,1]范围内！
    writer.add_images("input", imgs, global_step=step)
    # 将图像输入模型进行前向传播，得到Sigmoid激活后的输出
    output = tudui(imgs)
    # 将Sigmoid激活后的输出图像写入TensorBoard
    writer.add_images("output", output, step)
    # 步骤计数器加1，准备记录下一个批次
    step += 1

# 关闭SummaryWriter，确保所有数据写入磁盘
writer.close()