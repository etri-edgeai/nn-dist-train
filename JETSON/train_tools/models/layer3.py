import torch
import torch.nn as nn
import torch.nn.functional as F

__all__=['Layer3']


# only for synthetic benchmark
class Layer3(nn.Module):
    def __init__(self):
        super(Layer3, self).__init__()
        self.encoder = nn.Sequential(*[nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                       nn.Conv2d(64, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                                       nn.Conv2d(32, 16, 3, 2, 1), nn.BatchNorm2d(16)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)
        
    def forward(self, x, get_features=False):
        x = self.encoder(x)
        x = self.pool(x)
        out = x.view(x.size(0), -1)
        
        out = self.fc(out)
        
        if get_features:
            return out, x
        else:
            return out