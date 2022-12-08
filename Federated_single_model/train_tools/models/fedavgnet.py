import torch
import torch.nn as nn


class FedAvgNetMNIST(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    Model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, num_classes=10):
        super(FedAvgNetMNIST, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.ndim < 4:
            x = torch.unsqueeze(x, 1)
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class FedAvgNetCIFAR(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetCIFAR, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x


class FedAvgNetCIFARAdd(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetCIFARAdd, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.linear_2 = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.relu(self.linear_1(x))
        z = self.relu(self.linear_2(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x


class FedAvgNetTiny(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetTiny, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(16384, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x


class FedAvgNetCIFAR_BN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(FedAvgNetCIFAR_BN, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear_2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(self.bn1(x))
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(self.bn2(x))
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(self.bn3(x))
        x = self.linear_2(x)

        return x


class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 84)
        self.classifier = nn.Linear(84, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2)

    def forward(self, x, get_features=False):
        x = self.maxpool(self.relu(self.conv1(x)))
        x = self.maxpool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        z = self.relu(self.fc3(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x


class MotivCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(MotivCNN, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 2)
        self.classifier = nn.Linear(2, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.tanh(self.linear_1(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x


class MotivCNNAdd(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(MotivCNNAdd, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.linear_2 = nn.Linear(512, 2)
        self.classifier = nn.Linear(2, num_classes)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.tanh(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.tanh(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        x = self.tanh(self.linear_1(x))
        z = self.tanh(self.linear_2(x))
        x = self.classifier(z)

        if get_features:
            return x, z

        else:
            return x
