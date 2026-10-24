import torch
from Net import Net
from torch.autograd import Variable
from torchvision import transforms, datasets
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.datasets import ImageFolder  
from torch.utils.data import DataLoader

Root_Train = r'G:\01Git\AlexNet\Data\train'
Root_Test = r'G:\01Git\AlexNet\Data\val'

# 将图像像素归一化【-1,1】
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 归一化值 = （x-mean）/std

train_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 将图像缩放到指定大小
    transforms.RandomHorizontalFlip(), # 随机水平翻转
    transforms.ToTensor(), # 将图片转换为张量
    normalize # 归一化
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)), # 将图像缩放到指定大小
    transforms.ToTensor(), # 将图片转换为张量
    normalize # 归一化
])

test_dataset = ImageFolder(root=Root_Test, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

# 加载模型
model.load_state_dict(torch.load(r'G:\01Git\AlexNet\save_model\best_model.pth'))

classes = ['cat', 'dog']

# Tensor to PIL Image
show = ToPILImage()

# 验证阶段
model.eval() # 设置模型为评估模式
for i in range(3000):
    x, y = test_dataset[i] # 获取第i张图片和标签
    # show(x).show() # 显示图片
    x = x.unsqueeze(0) # 增加一个维度，变为[1, C, H, W]
    x = torch.tensor(x).to(device)
    with torch.no_grad(): # 不计算梯度
        output = model(x) # 前向传播
        _, pred = torch.max(output, 1) # 获取预测结果
        print("Predicted: {}, Actual: {}".format(classes[pred.item()], classes[y])) # 输出预测类别和真实类别
