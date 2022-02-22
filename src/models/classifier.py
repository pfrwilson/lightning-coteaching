
import math
import torch
from torch import nn
from torch.nn.functional import relu


class CNN(nn.Module):

    def __init__(self):
        
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(8)

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(4)

        self.pool = nn.MaxPool2d(kernel_size=4)

        self.flatten = nn.Flatten()

        self.fc = nn.Linear(4 * 7 * 7, 10)

    def forward(self, x):

        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))
        x = relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x