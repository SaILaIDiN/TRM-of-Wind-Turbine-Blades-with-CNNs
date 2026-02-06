import torch
import torch.nn as nn


class WindTurbineModel(nn.Module):
    def __init__(self, n_metadata, n_classes=3):
        super(WindTurbineModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 99 * 31, 128)
        self.fc2 = nn.Linear(128 + n_metadata, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x, metadata=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 99 * 31)
        x = self.fc1(x)
        x = self.relu(x)
        if metadata is not None:
            x = torch.cat((x, metadata), dim=1)
            # print("Valid Metadata.")
        else:
            x = x
            # print("Invalid Metadata.")
        x = self.fc2(x)
        return x
