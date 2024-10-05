import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

        
'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
class CustomHardtanh(nn.Module):
    def __init__(self, initial_param=1):
            super(CustomHardtanh, self).__init__()
            self.param = nn.Parameter(torch.tensor(float(initial_param), requires_grad=True))

    def forward(self, x):
        abs_param = torch.abs(self.param)
        return F.relu(x - abs_param) + abs_param - F.relu(x + abs_param)

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out
    
class Final_Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Final_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out
    

class MobileNetCifar(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100):
        super(MobileNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        return out
    
    
class MobileNetCifar_Sphere(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, norm=1):
        super(MobileNetCifar_Sphere, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(1024, num_classes, bias=False)
        self.norm=norm

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        return normalized_feature


class MobileNetCifar_ETF(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, norm=1):
        super(MobileNetCifar_ETF, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.projection= nn.Linear(1024, num_classes, bias=False)
        self.classifier = nn.Linear(num_classes, num_classes, bias=False)

        self.norm = nn.Parameter(torch.tensor([norm], dtype=torch.float))

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        out= self.projection(out)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        out= self.projection(out)
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        return normalized_feature
    
    
class MobileNetCifar_NCP(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, norm=1):
        super(MobileNetCifar_NCP, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(1024, num_classes, bias=False)
        self.norm=norm
        self.custom_hardtanh = CustomHardtanh()

    def _make_layers(self, in_planes):
        layers = []
        for i, x in enumerate(self.cfg):
            if i == len(self.cfg) - 1:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(Final_Block(in_planes, out_planes, stride))
                in_planes = out_planes

            else:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(Block(in_planes, out_planes, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.custom_hardtanh(out)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.custom_hardtanh(out)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)
        
        return normalized_feature
    
class MobileNetCifar_DR(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, norm=1):
        super(MobileNetCifar_DR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(1024, num_classes, bias=False)
        self.norm=norm

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)
        
        return normalized_feature

    def extract_original_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        
        return out

class MobileNettiny(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=200):
        super(MobileNettiny, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(4096, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        return out
    
    
class MobileNettiny_Sphere(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=200, norm=1):
        super(MobileNettiny_Sphere, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(4096, num_classes, bias=False)
        self.norm=norm

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        return normalized_feature


class MobileNettiny_ETF(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=200, norm=1):
        super(MobileNettiny_ETF, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.projection= nn.Linear(4096, num_classes, bias=False)
        self.classifier = nn.Linear(num_classes, num_classes, bias=False)

        self.norm = nn.Parameter(torch.tensor([norm], dtype=torch.float))

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        out= self.projection(out)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        out= self.projection(out)
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        return normalized_feature
    
    
class MobileNettiny_NCP(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=200, norm=1):
        super(MobileNettiny_NCP, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(4096, num_classes, bias=False)
        self.norm=norm
        self.custom_hardtanh = CustomHardtanh()

    def _make_layers(self, in_planes):
        layers = []
        for i, x in enumerate(self.cfg):
            if i == len(self.cfg) - 1:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(Final_Block(in_planes, out_planes, stride))
                in_planes = out_planes

            else:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(Block(in_planes, out_planes, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.custom_hardtanh(out)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.custom_hardtanh(out)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)
        
        return normalized_feature

class MobileNettiny_DR(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, norm=1):
        super(MobileNettiny_DR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(1024, num_classes, bias=False)
        self.norm=norm

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)
        
        return normalized_feature

    def extract_original_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        
        
        return out

    

    
class MobileNetImageNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100):
        super(MobileNetImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        
        return out
    
    
class MobileNetImageNet_Sphere(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, norm=1):
        super(MobileNetImageNet_Sphere, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(1024, num_classes, bias=False)
        self.norm=norm

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        return normalized_feature


class MobileNetImageNet_ETF(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, norm=1):
        super(MobileNetImageNet_ETF, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.projection= nn.Linear(1024, num_classes, bias=False)
        self.classifier = nn.Linear(num_classes, num_classes, bias=False)

        self.norm = nn.Parameter(torch.tensor([norm], dtype=torch.float))

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        
        out= self.projection(out)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        
        out= self.projection(out)
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        return normalized_feature
    
    
class MobileNetImageNet_NCP(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, norm=1):
        super(MobileNetImageNet_NCP, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(1024, num_classes, bias=False)
        self.norm=norm
        self.custom_hardtanh = CustomHardtanh()

    def _make_layers(self, in_planes):
        layers = []
        for i, x in enumerate(self.cfg):
            if i == len(self.cfg) - 1:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(Final_Block(in_planes, out_planes, stride))
                in_planes = out_planes

            else:
                out_planes = x if isinstance(x, int) else x[0]
                stride = 1 if isinstance(x, int) else x[1]
                layers.append(Block(in_planes, out_planes, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.custom_hardtanh(out)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.custom_hardtanh(out)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)
        
        return normalized_feature
    
class MobileNetImageNet_DR(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=100, norm=1):
        super(MobileNetImageNet_DR, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False)
        self.layers = self._make_layers(in_planes=32)
        self.classifier = nn.Linear(1024, num_classes, bias=False)
        self.norm=norm

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)#feature normalize!!
        
        logits=self.classifier(normalized_feature*self.norm)
        
        return logits
    
    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        
        normalized_feature= nn.functional.normalize(out, p=2, dim=1)
        
        return normalized_feature

    def extract_original_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        
        
        return out    