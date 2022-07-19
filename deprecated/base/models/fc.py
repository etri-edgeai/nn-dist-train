import torch
import torch.nn as nn
import torch.nn.functional as F

__all__=['FC']


# only for synthetic benchmark
class FC(nn.Module):
    def __init__(self, in_channels=60, num_classes=10, hidden_dim=500):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        return x