import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MnistCNN", "CifarCNN"]


class MnistCNN(nn.Module):
    def __init__(self, dim_in=3, num_classes=10, use_bias=True):
        super(MnistCNN, self).__init__()
        self.dim_in = dim_in
        self.conv1 = nn.Conv2d(dim_in, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(1024, 512)
        self.classifier = nn.Linear(512, num_classes, bias=use_bias)
        self.relu = nn.ReLU()
        print("MnistCNN was made")

    def forward(self, x, get_features=False):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.classifier(features)

        if get_features:
            return x, features

        else:
            return x


class CifarCNN(nn.Module):
    def __init__(
        self, dim_in=3, num_classes=10, use_bias=True, ortho_init=False, dim2_rep=False
    ):
        super(CifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, num_classes, bias=use_bias)

        if dim2_rep:
            self.fc2 = nn.Linear(512, 2)
            self.classifier = nn.Linear(2, num_classes, bias=use_bias)

        if ortho_init:
            torch.nn.init.orthogonal_(self.classifier.weight)

        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(2)
        # print("CifarCNN was made")

    def forward(self, x, get_features=False, get_grad=False):
        x = self.mp(self.relu(self.conv1(x)))
        x = self.mp(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        features = self.relu(self.fc2(x))

        if (features.requires_grad is True) and (get_grad is True):
            features.retain_grad()

        x = self.classifier(features)

        if get_features:
            return x, features

        else:
            return x
