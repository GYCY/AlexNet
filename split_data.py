import os
from shutil import copy
import random

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 获取Data下所有文件夹名
file_path = r'G:\01Git\AlexNet\data_name'
animal_classes = os.listdir(file_path)

# 创建训练集和验证集文件夹
mkdir('Data/train')
mkdir('Data/val')
mkdir('Data/test')
for class_name in animal_classes:
    mkdir(os.path.join('Data/train', class_name))
    mkdir(os.path.join('Data/val', class_name))
    mkdir(os.path.join('Data/test', class_name))

# 划分比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 遍历每个类别的文件夹
for class_name in animal_classes:
    class_path = os.path.join(file_path, class_name)
    images = os.listdir(class_path)
    random.shuffle(images)  # 打乱图片顺序

    # 计算训练集、验证集和测试集的数量
    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)
    test_count = int(len(images) * test_ratio)

    # 将图片划分到训练集、验证集和测试集中
    for i, image in enumerate(images):
        src_path = os.path.join(class_path, image)
        if i < train_count:
            dst_path = os.path.join('Data/train', class_name, image)
        elif i < train_count + val_count:
            dst_path = os.path.join('Data/val', class_name, image)
        else:
            dst_path = os.path.join('Data/test', class_name, image)
        copy(src_path, dst_path)

        print("\r[{}] processing class: {}/{}".format(class_name, i + 1, len(images)), end='')