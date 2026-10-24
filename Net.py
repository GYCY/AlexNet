import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)
        self.ReLU = nn.ReLU()
        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 3 2
        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 6 * 6, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=1000)
        self.fc4 = nn.Linear(in_features=1000, out_features=2)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s3(x)
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc4(x)
        return x

if __name__ == "__main__":
    model = Net()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output.shape)