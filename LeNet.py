import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1: conv with same padding
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)  # 'same' padding
        # S2: average pooling
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: conv, valid padding (no padding)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)  # default padding=0
        # S4: average pooling
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C5: conv, valid padding (no padding)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))   # C1
        x = self.pool1(x)               # S2
        x = torch.tanh(self.conv2(x))   # C3
        x = self.pool2(x)               # S4
        x = torch.tanh(self.conv3(x))   # C5
        x = x.view(x.size(0), -1)       # Flatten
        x = torch.tanh(self.fc1(x))     # FC6
        x = self.fc2(x)                 # FC7
        return F.log_softmax(x, dim=1)  # Softmax (log) output

model = LeNet5()
s = summary(model, (1, 1, 28, 28), col_names=['input_size', 'output_size', 'num_params', 'trainable'], row_settings=['var_names'])
