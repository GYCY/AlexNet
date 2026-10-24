# Python 3.8 GPU 环境

本项目使用 `.venv` 虚拟环境，基于本机的 64 位 Python 3.8.8。
依赖为 PyTorch 2.4.1、torchvision 0.19.1，使用 CUDA 12.1 构建。
绘图使用 Matplotlib 3.7.5，已加入 `requirements.txt`。

已于 2026-10-24 在 NVIDIA GeForce RTX 3060 Laptop GPU 上验证：
`torch.cuda.is_available()` 为 `True`，GPU 卷积、反向传播和
`torchvision.ops.nms` CUDA 运算均通过，`pip check` 未发现依赖冲突。

## VS Code

使用 VS Code 打开本项目文件夹。项目配置已将默认解释器指向
`.venv\Scripts\python.exe`。
如果 VS Code 已经选过其他解释器，按 `Ctrl+Shift+P`，运行
`Python: Select Interpreter`，选择本项目 `.venv\Scripts\python.exe`。

## 运行

在项目根目录的 PowerShell 中，无需激活环境即可直接运行：

```powershell
.\.venv\Scripts\python.exe Net.py
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

训练时，需要把模型和输入数据都放到 GPU：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
images = images.to(device)
labels = labels.to(device)
```

## 重建环境

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip==25.0.1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

本机系统代理的 HTTPS 配置与 Python 3.8 下载存在兼容问题。
如果安装时遇到 `ProxyError`，并且本机代理仍运行于 `127.0.0.1:7897`，
可显式指定 HTTP 代理重试：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --proxy http://127.0.0.1:7897
```
