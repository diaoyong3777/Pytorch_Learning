"""
        掌握以下能力
        1、重写Dataset加载数据和对应的标签甚至还可以返回路径

        2、transform数据转换
        # Compose：依次执行里面的转换。Resize大小。ToTensor转为tensor格式。更多变换见transforms.py
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        img = transform(img) # __call__让对象能够直接像函数一样调用

        3、tensorboard查看日志记录面板
        writer = SummaryWriter("logs") # 创建日志
        writer.add_image('test', train_dataset[201]['img']) # 添加笔记(板块名称、内容)
        终端执行：tensorboard --logdir logs --port 6006

        4、扩展知识
        读取图片：PIL.Image.open、cv.imread、Totensor。对应格式：PIL、nparray、tensor
        图片按照 tensor 的存储顺序存成 C H W（通道，长，宽）图片为RGB为3通道
        Totensor():
                # 输入：PIL图像 或 numpy数组
                # 输出：PyTorch Tensor
                
                # 1. 数据类型转换：
                # PIL图像 (H×W×C, uint8[0-255]) → Tensor (C×H×W, float32[0.0-1.0])
                
                # 2. 维度重排：
                # Height × Width × Channels → Channels × Height × Width
                
                # 3. 数值范围转换：【除以255】
                # 整数 [0, 255] → 浮点数 [0.0, 1.0]
                
                # 但注意：这只是简单的缩放，不是统计意义上的Normalization！
        Normalize(mean, std)：【三通道对应三个数字】
                # 改变数据的统计分布：使得均值=0，标准差=1
                # 公式：normalized = (tensor - mean) / std
        【理解，以3个数为例：【(76,127,178)=>totensor=>(0.3,0.5,0.7)=>norm=>(-0.1,0,0.1)】，这样就便于后续学习】
"""


# 导入PyTorch数据集基础类和数据加载器
from torch.utils.data import Dataset, DataLoader
# PIL（Python Imaging Library，Python图像处理库）
from PIL import Image
# 导入操作系统接口
import os
# 导入图像变换工具
from torchvision import transforms
# 导入TensorBoard可视化工具
from torch.utils.tensorboard import SummaryWriter

# 创建TensorBoard日志记录器，日志保存到logs文件夹
writer = SummaryWriter("logs")


# 定义自定义数据集类
class MyData(Dataset): # MyData 继承自 Dataset
    """自定义数据集类，用于加载图像和标签数据"""

    # 初始化数据集
    def __init__(self, root_dir, image_dir, label_dir, transform):
        """
        初始化数据集
        root_dir: 数据集根目录路径
        image_dir: 图像文件所在子目录名
        label_dir: 标签文件所在子目录名
        transform: 图像预处理变换操作
        """
        self.root_dir = root_dir  # 存储根目录路径
        self.image_dir = image_dir  # 存储图像目录名
        self.label_dir = label_dir  # 存储标签目录名
        self.label_path = os.path.join(self.root_dir, self.label_dir)  # 拼接完整标签路径
        self.image_path = os.path.join(self.root_dir, self.image_dir)  # 拼接完整图像路径
        self.image_list = os.listdir(self.image_path)  # 获取图像文件列表 listdir:获取指定目录下的所有文件和子目录的名称列表
        self.label_list = os.listdir(self.label_path)  # 获取标签文件列表
        self.transform = transform  # 存储图像变换操作
        # 因为label和Image文件名相同，进行一样的排序，可以保证取出的数据和label是一一对应的
        self.image_list.sort()  # 对图像文件名排序
        self.label_list.sort()  # 对标签文件名排序

    # 获取单个数据样本
    def __getitem__(self, idx): # 当使用 dataset[index] 时自动执行
        """
        根据索引获取数据样本
        idx: 样本索引号
        返回: 包含图像和标签的字典
        数据流示例: idx=0 → 读取第一个图像和标签文件 → 返回{'img': tensor, 'label': 'ant'}
        """
        img_name = self.image_list[idx]  # 根据索引获取图像文件名
        label_name = self.label_list[idx]  # 根据索引获取标签文件名
        img_item_path = os.path.join(self.root_dir, self.image_dir, img_name)  # 拼接图像完整路径
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)  # 拼接标签完整路径
        img = Image.open(img_item_path)  # 打开图像文件

        with open(label_item_path, 'r') as f:  # 以只读方式打开标签文件
            label = f.readline()  # 读取标签内容（第一行）

        # img = np.array(img)  # 可选：将图像转为numpy数组（当前注释）
        img = self.transform(img)  # 对图像应用预处理变换
        sample = {'img': img, 'label': label ,'path': img_item_path}  # 创建样本字典
        return sample  # 返回样本数据

    # 获取数据集大小
    def __len__(self):  # 当调用 len(dataset) 时自动执行
        """返回数据集中样本的总数量"""
        # assert:检查某个条件是否为真，如果为假则抛出异常. assert 条件表达式, "可选的错误信息"
        assert len(self.image_list) == len(self.label_list),"图像和标签数量不匹配！"  # 确保图像和标签数量一致
        return len(self.image_list)  # 返回图像文件数量


# 程序主入口
if __name__ == '__main__':
    # 定义图像预处理流程：调整大小 → 转为Tensor
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    root_dir = "dataset/train"  # 数据集根目录路径
    image_ants = "ants_image"  # 蚂蚁图像目录名
    label_ants = "ants_label"  # 蚂蚁标签目录名
    # 创建蚂蚁数据集实例
    ants_dataset = MyData(root_dir, image_ants, label_ants, transform)
    image_bees = "bees_image"  # 蜜蜂图像目录名
    label_bees = "bees_label"  # 蜜蜂标签目录名
    # 创建蜜蜂数据集实例
    bees_dataset = MyData(root_dir, image_bees, label_bees, transform)
    # 合并两个数据集 这个+应该是Dataset类里面的运算符重载，你没继承的话肯定加不了
    train_dataset = ants_dataset + bees_dataset

    # 创建数据加载器，批量大小1，迭代器每次返回1个数据，使用主进程加载数据(多个会更快)
    dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0)

    # 将第0个样本的图像写入TensorBoard，标签为'test'
    print(train_dataset[201])
    writer.add_image('test', train_dataset[201]['img'])

    # 遍历数据加载器并打印信息
    for i, j in enumerate(dataloader):
        # index, {img, label, path} = j
        # print(j)
        print(i, j['img'].shape, j['label'], j['path'])
        writer.add_image("train_data", j['img'][0] , i)

    writer.close() # 关闭TensorBoard写入器
