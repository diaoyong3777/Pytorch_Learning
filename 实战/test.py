"""
    1、模型应用注意事项
    image = image.convert('RGB') 转为RGB格式（确保3个通道）
    model = torch.load("tudui_9.pth", weights_only=False ,map_location=torch.device('cpu'))
    【GPU跑的模型在CPU跑需要映射 map_location=torch.device('cpu')】
    image = torch.reshape(image, (1, 3, 32, 32)) # 添加 batch维度
    model.eval() # 设置模型为评估模式
    with torch.no_grad(): # 在推理时不计算梯度（节省内存和计算资源）
        output = model(image)  # 前向传播，获得预测结果

"""

# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from PIL import Image
from torch import nn

# 图像路径
image_path = "飞机.jpg"
# 打开图像文件
image = Image.open(image_path)
print(image)  # 打印图像基本信息（格式、尺寸等）
# 将图像转换为RGB格式（确保3个通道）
image = image.convert('RGB')

# 定义图像预处理变换管道
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),  # 调整尺寸为32x32
                                            torchvision.transforms.ToTensor()])  # 转换为Tensor格式

# 对图像应用预处理变换
image = transform(image)
print(image.shape)  # 打印变换后的图像张量形状


# 定义神经网络模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # 使用Sequential容器按顺序组织网络层
        self.model = nn.Sequential(
            # 第一个卷积层：输入通道3(RGB)，输出通道32，5x5卷积核，步长1，填充2（保持尺寸不变）
            nn.Conv2d(3, 32, 5, 1, 2),
            # 最大池化层：2x2窗口，步长2（尺寸减半）
            nn.MaxPool2d(2),
            # 第二个卷积层：输入32通道，输出32通道
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            # 第三个卷积层：输入32通道，输出64通道
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            # 展平层：将多维特征图展平为一维向量
            nn.Flatten(),
            # 全连接层：从64*4*4=1024个特征到64个神经元
            nn.Linear(64 * 4 * 4, 64),
            # 输出层：从64个神经元到10个类别（CIFAR-10数据集）
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 加载预训练模型（映射到CPU设备）
model = torch.load("tudui_9.pth", weights_only=False ,map_location=torch.device('cpu'))
print(model)  # 打印模型结构

# 调整图像张量形状：从(3,32,32)变为(1,3,32,32) - 添加batch维度
image = torch.reshape(image, (1, 3, 32, 32))

# 设置模型为评估模式（关闭dropout、batch normalization的训练特定行为）
model.eval()

# 在推理时不计算梯度（节省内存和计算资源）
with torch.no_grad():
    output = model(image)  # 前向传播，获得预测结果

print(output)  # 打印原始输出（10个类别的logits）

# 获取预测类别（找到最大值的索引）
print(output.argmax(1))  # 输出预测的类别编号

# 'airplane'= {int} 0
# 'automobile' = {int} 1
# 'bird'= [int} 2
# 'cat' = {int} 3
# 'deer' = {int} 4
# 'dog' = {[int} 5
# 'frog'= {int} 6
# 'horse'= {int} 7
# 'ship' = {int} 8
# 'truck'’= {int} 9