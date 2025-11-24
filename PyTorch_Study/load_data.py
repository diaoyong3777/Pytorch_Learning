"""
    1、官方文档：https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    2、数据处理流程：
    原始数据 → ToTensor转换 → DataLoader分批 → TensorBoard可视化
    test_data[0] → [3,32,32]张量 → 64个样本组成批次 → 在TensorBoard中显示
    3、add_images
    add_image：用于添加单张图像（3D张量：C×H×W）
    add_images：用于添加多张图像（4D张量：N×C×H×W），适合显示整个批次
    4、tensorboard --logdir="logs_dataloader" --samples_per_plugin=images=10000
"""

# 导入torchvision库，包含常用的数据集和图像转换
import torchvision

# 准备的测试数据集
# 导入DataLoader用于批量加载数据
from torch.utils.data import DataLoader
# 导入SummaryWriter用于TensorBoard可视化
from torch.utils.tensorboard import SummaryWriter

# 创建CIFAR-10测试数据集
# CIFAR-10包含10个类别的60000张32x32彩色图像，其中测试集10000张
test_data = torchvision.datasets.CIFAR10(
    "./dataset",  # 数据集存储路径
    train=False,  # 加载测试集（如果为True则加载训练集）
    transform=torchvision.transforms.ToTensor()  # 将图像转换为PyTorch张量，并归一化到[0,1]范围
)

# 创建数据加载器，用于批量加载测试数据
test_loader = DataLoader(
    dataset=test_data,  # 要加载的数据集对象
    batch_size=64,  # 每个批次包含64个样本
    shuffle=True,  # 每个epoch开始时打乱数据顺序
    num_workers=0,  # 使用0个工作进程（在主进程中加载数据）
    drop_last=True  # 丢弃最后一个不完整的批次（如果样本数不能被batch_size整除）
)

# 测试数据集中第一张图片及target（标签）
# 从测试数据集中获取第一个样本
img, target = test_data[0]
# 打印图像张量的形状：对于CIFAR-10是[3, 32, 32]（通道数×高度×宽度）
print(img.shape)
# 打印标签：一个0-9的整数，代表10个类别中的某一个
print(target)

# 创建TensorBoard的SummaryWriter对象，日志将保存在"logs_dataloader"目录中
writer = SummaryWriter("logs_dataloader")

# 循环2个epoch（虽然测试集通常只需要1个epoch，这里演示多epoch的数据加载）
for epoch in range(2):
    step = 0  # 初始化步骤计数器，用于TensorBoard中的全局步数
    # 遍历数据加载器，每次返回一个批次的数据
    for data in test_loader:
        # 解包批次数据：imgs包含64张图像，targets包含64个标签
        imgs, targets = data
        # 以下是被注释掉的调试代码：
        # print(imgs.shape)  # 打印批次图像形状：[64, 3, 32, 32]（批次大小×通道数×高度×宽度）
        # print(targets)     # 打印批次标签：包含64个0-9整数的张量

        # 将整个批次的图像添加到TensorBoard中
        # "Epoch: {}".format(epoch): 图像标签，包含epoch信息便于区分
        # imgs: 批次图像张量，形状为[64, 3, 32, 32]
        # step: 当前步骤，用于在TensorBoard中区分不同批次的图像
        writer.add_images("Epoch: {}".format(epoch), imgs, step)

        # 被注释的替代写法：显式指定数据格式
        # writer.add_image(tag="batched_img", img_tensor=imgs, global_step=step, dataformats="NCHW")

        step = step + 1  # 步骤计数器加1，准备记录下一个批次

# 关闭SummaryWriter，确保所有数据都写入磁盘
writer.close()
