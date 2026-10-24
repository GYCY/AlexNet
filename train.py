import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from Net import Net
import numpy as np
import os

from torchvision import transforms, datasets # 对图片做预处理; 提供数据集相关工具，包括读取本地图片
from torchvision.datasets import ImageFolder # 按照文件夹结构读取图片，并自动分配类别标签
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

Root_Train = r'G:\01Git\AlexNet\Data\train'
Root_Val = r'G:\01Git\AlexNet\Data\val'

# 将图像像素归一化【-1,1】
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化值 = （x-mean）/std

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 将图像缩放到指定大小
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.ToTensor(), # 将图片转换为张量
    normalize # 归一化
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 将图像缩放到指定大小
    transforms.ToTensor(), # 将图片转换为张量
    normalize # 归一化
])

train_dataset = ImageFolder(root=Root_Train, transform=train_transform)
val_dataset = ImageFolder(root=Root_Val, transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # SGD优化器
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # 学习率衰减

# 训练函数
def train(model, dataloader, criterion, optimizer):
    loss, current, total = 0.0, 0.0, 0
    model.train() # 设置模型为训练模式
    for batch, (x,y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        output = model(x) # 前向传播
        cur_loss = criterion(output, y) # 计算损失
        _, pred = torch.max(output, 1) # 获取预测结果
        cur_acc = torch.sum(pred == y).item() / y.size(0) # 计算当前批次的准确率

        optimizer.zero_grad() # 梯度清零
        cur_loss.backward() # 反向传播
        optimizer.step() # 更新参数

        loss += cur_loss.item()
        current += cur_acc
        total += 1

    train_loss = loss / total
    train_acc = current / total
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.4f}%")
    return train_loss, train_acc

# 验证函数
def val(model, dataloader, criterion):
    loss, current, total = 0.0, 0.0, 0
    model.eval() # 设置模型为评估模式
    with torch.no_grad(): # 不计算梯度
        for batch, (x,y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x) # 前向传播
            cur_loss = criterion(output, y) # 计算损失
            _, pred = torch.max(output, 1) # 获取预测结果
            cur_acc = torch.sum(pred == y).item() / y.size(0) # 计算当前批次的准确率

            loss += cur_loss.item()
            current += cur_acc
            total += 1

    val_loss = loss / total
    val_acc = current / total
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.4f}%")
    return val_loss, val_acc

# 画图函数
def plot_loss_acc(loss_train, acc_train, loss_val, acc_val):
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "save_model")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(loss_train, label='Train Loss')
    plt.plot(loss_val, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.title('Loss Curve')
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(acc_train, label='Train Acc')
    plt.plot(acc_val, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.title('Accuracy Curve')
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()

# 开始训练
loss_train, acc_train, loss_val, acc_val = [], [], [], []
num_epochs = 20
max_acc = 0.0
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}------------------------------")
    train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
    val_loss, val_acc = val(model, val_dataloader, criterion)
    lr_scheduler.step() # 每轮训练完成后更新学习率，确保在 optimizer.step() 之后

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    # 保存模型
    if val_acc > max_acc:
        folder = "save_model"
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        max_acc = val_acc
        torch.save(model.state_dict(), "save_model/best_model.pth")
        print(f"Best model saved with accuracy: {max_acc*100:.4f}%")

print("Training completed.")
plot_loss_acc(loss_train, acc_train, loss_val, acc_val)
