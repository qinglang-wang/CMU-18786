import torch
import torch.nn as nn
import torch.nn.functional as F
from mytorch import MyConv2D, MyMaxPool2D


class FCNN(nn.Module):
    def __init__(self, input_dim=3*32*32, hidden_dims=[512, 256], num_classes=100, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


class CNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.backbone = nn.Sequential(
            MyConv2D(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            MyConv2D(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            MyMaxPool2D(kernel_size=2, stride=2),

            MyConv2D(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            MyMaxPool2D(kernel_size=2, stride=2),

            MyConv2D(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x.view(x.size(0), -1))
        return x
