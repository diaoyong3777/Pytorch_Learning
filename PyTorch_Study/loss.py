"""
    1、了解三种损失函数
    nn.L1Loss(差的绝对值求和)、nn.MSELoss(均方损失)、nn.CrossEntropyLoss(交叉熵损失)

    2、nn.CrossEntropyLoss(交叉熵损失) <=> softmax +  CrossEntropyLoss

    3、回归(异常值不敏感，抗噪声)、回归(异常值敏感，强调精度)、分类

    4、利用损失函数从预测值开始反向传播 loss(predict, target).backward()
    requires_grad=True启用梯度计算(tensor里自带了grad属性)(单独的tensor默认不开启，神经网络默认开启)
"""


# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
# 导入PyTorch深度学习框架
import torch
# 从神经网络模块中导入L1Loss损失函数
from torch.nn import L1Loss
# 从PyTorch导入神经网络模块
from torch import nn

# 创建输入张量和目标张量，用于演示损失函数计算
# 输入张量：模型的预测值
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
# 目标张量：真实标签值
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

# 重塑张量形状，使其符合神经网络输出的常见格式
# 形状从 [3] 变为 [1, 1, 1, 3] - 批次大小1，通道数1，高度1，宽度3
inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# 创建L1损失函数实例，也称为平均绝对误差(MAE)
# reduction='sum': 对所有元素的损失求和（而不是求平均）
loss = L1Loss(reduction='sum')
# 计算L1损失：|1-1| + |2-2| + |3-5| = 0 + 0 + 2 = 2
result = loss(inputs, targets)

# 创建MSE损失函数实例，也称为均方误差
loss_mse = nn.MSELoss()
# 计算MSE损失：[(1-1)² + (2-2)² + (3-5)²] / 3 = [0 + 0 + 4] / 3 ≈ 1.333
result_mse = loss_mse(inputs, targets)

# 打印L1损失和MSE损失的结果
print(result)      # 输出: tensor(2.)
print(result_mse)  # 输出: tensor(1.3333)


# 创建新的输入和标签，用于演示交叉熵损失
# 输入：3个类别的预测得分（logits），未经过softmax
x = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)  # 关键修改：启用梯度计算
# 目标标签：真实类别索引（这里类别1，索引从0开始）
y = torch.tensor([1])
# 重塑输入形状为 [1, 3] - 批次大小1，3个类别
# x = torch.reshape(x, (1, 3))
# 创建交叉熵损失函数实例，常用于多分类问题
loss_cross = nn.CrossEntropyLoss()
# 计算交叉熵损失(Loss = -x_class + log(∑e^x))
result_cross = loss_cross(x, y)
# 打印交叉熵损失结果
print(result_cross)

# 执行反向传播，计算梯度
# 这会计算损失函数相对于输入x的梯度：d(loss)/d(x)
result_cross.backward()

# 查看梯度
# 梯度表示：为了减少损失，每个输入值应该调整的方向和幅度
print("输入x的梯度:", x.grad)