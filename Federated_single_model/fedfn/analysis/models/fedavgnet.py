import torch
import torch.nn as nn



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

    def extract_features(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        
        return z
    
    
class FedAvgNetCIFAR_FN(torch.nn.Module):
    def __init__(self, num_classes=10, norm=1):
        super(FedAvgNetCIFAR_FN, self).__init__()
        self.conv2d_1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(4096, 512)
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.norm=norm

    def forward(self, x, get_features=False):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        
        normalized_feature= nn.functional.normalize(z, p=2, dim=1)#feature normalize!!
        
        logits = self.classifier(normalized_feature*self.norm)

        if get_features:
            return logits, z

        else:
            return logits
    def extract_features(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.flatten(x)
        z = self.relu(self.linear_1(x))
        normalized_feature= nn.functional.normalize(z, p=2, dim=1)#feature normalize!!
        
        return normalized_feature
