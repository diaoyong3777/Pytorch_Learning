"""
注释说明：
    1、完整训练流程：数据准备 -> 模型创建 -> 训练循环 -> 测试验证 -> 模型保存
    2、批次(batch) vs 轮次(epoch)：
       - 批次：一次处理的数据量(batch_size=64)
       - 轮次：完整遍历一遍训练集的次数(epoch=10)
    3、tudui.train()、tudui.eval()模式：【不管以后什么情景，都加上就行】
       - train(): 启用Dropout、BatchNorm等训练特有层
       - eval(): 关闭上述层，固定参数用于推理
    4、Tensorboard使用：浅色的的才是真实曲线，真实的曲线往往不好看，所以加入了平滑产生了颜色深的线
    5、with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源
    6、计算准确率：预测类别与真实类别一致的样本数
    for batch:
        correct = (outputs.argmax(1) == targets).sum()  # argmax(1)获取每行最大值的索引
        total_correct += correct
    acc = total_correct / test_data_size
"""

# -*- coding: utf-8 -*-
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model import *
from torch import nn
from torch.utils.data import DataLoader

# 1. 准备数据集
# CIFAR-10数据集：10个类别，32x32彩色图像
train_data = torchvision.datasets.CIFAR10(
    root="../PyTorch_Study/dataset",
    train=True,
    transform=torchvision.transforms.ToTensor(),  # 将PIL图像转为Tensor，并归一化到[0,1]
    download=True  # 如果数据集不存在则自动下载
)
test_data = torchvision.datasets.CIFAR10(
    root="../PyTorch_Study/dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 查看数据集大小
train_data_size = len(train_data)  # 训练集样本数：50000
test_data_size = len(test_data)  # 测试集样本数：10000
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 2. 创建数据加载器
# DataLoader负责批量加载数据，支持随机打乱、多进程等
train_dataloader = DataLoader(train_data, batch_size=64)  # 每批64个样本
test_dataloader = DataLoader(test_data, batch_size=64)

# 3. 创建网络模型
tudui = Tudui()  # 从model.py导入的自定义模型

# 4. 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失，适用于多分类问题

# 优化器：随机梯度下降
learning_rate = 1e-2  # 学习率0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)  # 传入模型参数和学习率

# 5. 设置训练参数
total_train_step = 0  # 记录总训练步数(批次数量)
total_test_step = 0  # 记录总测试次数
epoch = 10  # 训练轮数

# 6. 初始化Tensorboard用于可视化
writer = SummaryWriter("./logs_train")

# 7. 训练循环 - 核心部分
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练模式开始
    tudui.train()  # 设置模型为训练模式(启用Dropout等)

    # 遍历训练集的所有批次
    for data in train_dataloader:
        imgs, targets = data  # 解包数据：图像和标签

        # 前向传播
        outputs = tudui(imgs)  # 模型预测
        loss = loss_fn(outputs, targets)  # 计算损失

        # 反向传播
        optimizer.zero_grad()  # 清空之前的梯度
        loss.backward()  # 计算当前梯度
        optimizer.step()  # 更新模型参数

        # 记录训练信息
        total_train_step += 1
        if total_train_step % 100 == 0:  # 每100步打印一次
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试模式开始
    tudui.eval()  # 设置模型为评估模式(关闭Dropout等)
    total_test_loss = 0  # 累计测试损失
    total_accuracy = 0  # 累计正确预测数

    # 测试阶段不需要计算梯度
    with torch.no_grad():  # 关闭梯度计算，节省内存和计算资源
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)

            total_test_loss += loss.item()  # 累加批次损失

            # 计算准确率：预测类别与真实类别一致的样本数
            accuracy = (outputs.argmax(1) == targets).sum()  # argmax(1)获取每行最大值的索引
            total_accuracy += accuracy

    # 输出测试结果
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))

    # 记录到Tensorboard
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 8. 保存模型
    torch.save(tudui, "tudui_{}.pth".format(i))  # 保存整个模型
    print("模型已保存")

# 关闭Tensorboard写入器
writer.close()