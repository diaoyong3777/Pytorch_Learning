"""
    1、构建池化层
    from torch.nn import MaxPool2d
    self.maxpool1 = MaxPool2d(kernel_size=3)
    2、输入是四维的(批次、通道、高、宽)。可以通过reshape处理

    - 最大池化的好处：
    降低维度：减少参数数量和计算量
    保持特征：保留最显著的特征
    防止过拟合：提供一定的平移不变性
    扩大感受野：使后续层能看到更广的区域
"""

# 导入PyTorch深度学习框架
import torch
# 导入torchvision，包含计算机视觉相关的数据集和变换
import torchvision
# 从PyTorch导入神经网络模块
from torch import nn
# 从神经网络模块中导入二维最大池化层
from torch.nn import MaxPool2d
# 导入数据加载器
from torch.utils.data import DataLoader
# 导入TensorBoard可视化工具
from torch.utils.tensorboard import SummaryWriter

# 加载CIFAR-10测试数据集
dataset = torchvision.datasets.CIFAR10(
    "./dataset",  # 数据集存储路径
    train=False,  # 使用测试集（10000张图像）
    download=True,  # 如果数据集不存在则自动下载
    transform=torchvision.transforms.ToTensor()  # 将图像转换为PyTorch张量，归一化到[0,1]
)

# 创建数据加载器，用于批量加载数据
dataloader = DataLoader(
    dataset,  # 要加载的数据集
    batch_size=64  # 每个批次包含64个样本
)


# 定义自定义神经网络模型Tudui
class Tudui(nn.Module):
    """
    池化层演示模型
    功能：展示最大池化层的作用和特征图变化
    """

    def __init__(self):
        """
        初始化模型，定义池化层
        """
        # 调用父类nn.Module的初始化方法
        super(Tudui, self).__init__()
        # 定义二维最大池化层
        self.maxpool1 = MaxPool2d(
            kernel_size=3,  # 池化窗口大小：3x3
            ceil_mode=False  # 向下取整模式：当剩余区域不足时丢弃
        )

    def forward(self, input):
        """
        前向传播过程
        input: 输入张量，形状为 [batch_size, 3, 32, 32]
        返回: 池化后的特征图

        数据流示例：
        输入: [64, 3, 32, 32] → 最大池化 → 输出: [64, 3, 10, 10]
        计算: (32 - 3) / 3 + 1 = 10 （当ceil_mode=False时）
        """
        # 将输入通过最大池化层
        output = self.maxpool1(input)
        # 返回池化结果
        return output


# 创建模型实例
tudui = Tudui()

# 创建TensorBoard的SummaryWriter，日志保存在"logs_maxpool"文件夹中
writer = SummaryWriter("logs_maxpool")

# 初始化步骤计数器，用于TensorBoard中的全局步数
step = 0

# 遍历数据加载器，处理每个批次的数据
for data in dataloader:
    # 解包批次数据：imgs为图像张量，targets为标签
    imgs, targets = data
    # 将原始输入图像写入TensorBoard
    writer.add_images("input", imgs, step)
    # 将图像输入模型进行前向传播，得到池化后的输出
    output = tudui(imgs)
    # 将池化后的输出图像写入TensorBoard
    writer.add_images("output", output, step)
    # 步骤计数器加1，准备记录下一个批次
    step = step + 1

# 关闭SummaryWriter，确保所有数据写入磁盘
writer.close()