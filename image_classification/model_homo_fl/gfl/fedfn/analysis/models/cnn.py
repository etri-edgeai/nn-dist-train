import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        logits = self.classifier(features)
        return logits    
    
    def extract_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        
        return features
    
    
class CNNCifar_FN(nn.Module):
    def __init__(self, args, norm=1):
        super(CNNCifar_FN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, args.num_classes)
        self.norm=norm

    def forward(self, x):
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!
        
        logits = self.classifier(normalized_feature*self.norm)
        
        return logits    
        
    
    def extract_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        
        return normalized_feature
    
    