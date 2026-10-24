# AlexNet 猫狗图像分类

使用 PyTorch 实现的 AlexNet 风格二分类学习项目，包含数据划分、模型训练、验证、权重保存和逐张图片预测。输入为 RGB 图片，统一缩放至 `224 × 224`，输出为猫、狗两个类别的分数。

本项目的通道数、全连接层和激活函数配置以 [Net.py](Net.py) 为准，是对 AlexNet 的修改实现。

## 项目结构

```text
AlexNet/
├── Net.py                     # 网络结构
├── split_data.py              # 按类别划分训练集、验证集和测试集
├── train.py                   # 训练、验证、保存最佳权重和曲线
├── test.py                    # 加载权重并逐张打印预测结果
├── requirements.txt           # Python 依赖
├── ENVIRONMENT.md             # 本机 GPU 环境及 VS Code 配置说明
├── AlexNet.jpg                # 网络结构参考图
├── data_name/                 # 原始图片，按类别放入子文件夹（自行准备）
├── Data/                      # 划分后的数据（自行准备）
│   ├── train/
│   ├── val/
│   └── test/
└── save_model/
    ├── best_model.pth         # 验证准确率最高时的模型参数
    ├── loss_curve.png         # 训练集、验证集损失曲线
    └── accuracy_curve.png     # 训练集、验证集准确率曲线
```

`data_name/`、`Data/` 和 `.venv/` 已加入 `.gitignore`，克隆仓库后需要自行准备数据和 Python 环境。

## 安装环境

当前项目使用以下版本：

| 组件 | 版本 |
| --- | --- |
| Python | 3.8，本机为 3.8.8（64 位） |
| PyTorch | 2.4.1+cu121 |
| torchvision | 0.19.1+cu121 |
| CUDA 运行时 | 12.1 |
| NumPy | 1.24.4 |
| Matplotlib | 3.7.5 |

在 Windows PowerShell 中进入项目根目录，然后执行：

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip==25.0.1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

检查 PyTorch 和 GPU：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

本机已在 NVIDIA GeForce RTX 3060 Laptop GPU 上验证 GPU 运算。VS Code 解释器选择及代理下载问题见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 准备数据

将原始图片按类别整理：

```text
data_name/
├── Cat/
│   ├── 0.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    └── ...
```

代码当前使用本机绝对路径 `G:\01Git\AlexNet`。如果项目位于其他位置，运行前修改以下路径：

| 文件 | 需要检查的路径 |
| --- | --- |
| `split_data.py` | `file_path`：原始图片目录 |
| `train.py` | `Root_Train`、`Root_Val`：训练集和验证集目录 |
| `test.py` | `Root_Test`：预测数据目录；`torch.load(...)`：权重文件路径 |

在项目根目录运行数据划分脚本：

```powershell
.\.venv\Scripts\python.exe split_data.py
```

脚本会复制图片，按每个类别约 **80% / 10% / 10%** 的比例划分为 `Data/train`、`Data/val` 和 `Data/test`，余数归入测试集。首次划分时应使用空的 `Data` 输出目录；已有划分可直接复用，避免反复随机复制导致不同子集混入相同图片。

`ImageFolder` 根据类别文件夹名称排序生成标签，训练、验证和测试目录应包含相同的类别子文件夹。当前 `test.py` 的显示名称为 `['cat', 'dog']`，应与训练时的类别顺序一致。

## 训练和验证

```powershell
.\.venv\Scripts\python.exe train.py
```

当前训练设置：

- 输入处理：缩放到 `224 × 224`，转为张量，使用 `mean=[0.5, 0.5, 0.5]`、`std=[0.5, 0.5, 0.5]` 归一化。
- 数据增强：仅训练集使用随机水平翻转。
- 批次大小：32；训练集打乱顺序，验证集保持顺序。
- 训练轮数：20。
- 损失函数：交叉熵 `CrossEntropyLoss`。
- 优化器：SGD，初始学习率 `0.01`，动量 `0.9`。
- 学习率调度：每完成 10 轮训练，学习率乘以 `0.5`。
- 设备：CUDA 可用时优先使用 GPU。

每轮打印训练和验证的损失、准确率；验证准确率超过此前最佳值时，将模型参数保存到 `save_model/best_model.pth`。请从项目根目录启动训练，以使权重保存到项目内。

训练结束后绘制两张曲线，以 300 DPI 保存到项目的 `save_model/`。每张图在 `plt.show()` 前保存；普通脚本模式下，关闭第一张图窗后才会继续生成、保存第二张图。

当前损失与准确率是各批次指标的算术平均，最后一批不足 32 张时也占一个批次的权重。

## 加载模型并预测

`test.py` 会加载 `best_model.pth`，逐张输出预测类别和真实类别。**当前代码的 `Root_Test` 指向 `Data/val`，实际使用的是验证集。** 要评估最终模型对独立测试集的表现，请将其改为 `Data/test`。

脚本当前固定遍历 `range(3000)`。运行前，应按数据集大小将循环改为：

```python
for i in range(len(test_dataset)):
```

确认数据路径、权重路径和循环范围后运行：

```powershell
.\.venv\Scripts\python.exe test.py
```

脚本使用 `model.eval()` 和 `torch.no_grad()` 进行预测，当前只打印逐张结果，没有汇总整个测试集的准确率。训练集用于学习参数，验证集用于选择最佳权重，独立测试集用于最终评估。

## 检查网络输入输出

```powershell
.\.venv\Scripts\python.exe Net.py
```

该命令用一张随机 RGB 图片检查前向传播，预期输出形状为：

```text
torch.Size([1, 2])
```

其中 `1` 是批次大小，`2` 是类别数。
