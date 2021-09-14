import torch
import torch.nn as nn
import torch.nn.functional as F

__all__=['LeNet', 'LeNetContainer']


class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        if num_classes == 10 or num_classes == 100:
            self.fc1 = nn.Linear(5*5*50, 500)
        elif num_classes == 62: # federated emnist
            self.fc1 = nn.Linear(4*4*50, 500)
        else: # landmark
            self.fc1 = nn.Linear(53*53*50, 500)
        self.fc2 = nn.Linear(500, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x

    
class LeNetContainer(nn.Module):
    def __init__(self, num_filters, kernel_size, input_dim, hidden_dims, in_channels=1, num_classes=10):
        super(LeNetContainer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, num_filters[0], kernel_size, 1)
        self.conv2 = nn.Conv2d(num_filters[0], num_filters[1], kernel_size, 1)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x