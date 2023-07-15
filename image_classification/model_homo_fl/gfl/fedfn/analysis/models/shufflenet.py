import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision

from functools import partial
    
#  ############################################################################################################
# # SHUFFLENET
# ############################################################################################################



# class ShuffleBlock(nn.Module):
#     def __init__(self, groups):
#         super(ShuffleBlock, self).__init__()
#         self.groups = groups

#     def forward(self, x):
#         '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
#         N,C,H,W = x.size()
#         g = self.groups
#         return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


# class Bottleneck(nn.Module):
#     def __init__(self, in_planes, out_planes, stride, groups):
#         super(Bottleneck, self).__init__()
#         self.stride = stride

#         mid_planes = out_planes//4
#         g = 1 if in_planes==24 else groups
#         self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
#         self.bn1 = nn.BatchNorm2d(mid_planes)
#         self.shuffle1 = ShuffleBlock(groups=g)
#         self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
#         self.bn2 = nn.BatchNorm2d(mid_planes)
#         self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_planes)

#         self.shortcut = nn.Sequential()
#         if stride == 2:
#             self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.shuffle1(out)
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         res = self.shortcut(x)
#         out = F.relu(torch.cat([out,res], 1)) if self.stride==2 else F.relu(out+res)
#         return out


# class ShuffleNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ShuffleNet, self).__init__()
#         cfg = {'out_planes': [200,400,800],'num_blocks': [4,8,4],'groups': 2}

#         out_planes = cfg['out_planes']
#         num_blocks = cfg['num_blocks']
#         groups = cfg['groups']

#         self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(24)
#         self.in_planes = 24
#         self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
#         self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
#         self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
#         self.classifier = nn.Linear(out_planes[2], num_classes)
     

#     def _make_layer(self, out_planes, num_blocks, groups):
#         layers = []
#         for i in range(num_blocks):
#             stride = 2 if i == 0 else 1
#             cat_planes = self.in_planes if i == 0 else 0
#             layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
#             self.in_planes = out_planes
#         return nn.Sequential(*layers)


#     def extract_features(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, 4)
#         feature = out.view(out.size(0), -1)
#         return feature


#     def forward(self, x):
#         feature = self.extract_features(x)
#         out = self.classifier(feature)
#         return out   
    
    
# class ShuffleNet_FN(nn.Module):
#     def __init__(self, num_classes=10, norm=1):
#         super(ShuffleNet_FN, self).__init__()
#         cfg = {'out_planes': [200,400,800],'num_blocks': [4,8,4],'groups': 2}

#         out_planes = cfg['out_planes']
#         num_blocks = cfg['num_blocks']
#         groups = cfg['groups']

#         self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(24)
#         self.in_planes = 24
#         self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
#         self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
#         self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
#         self.classifier = nn.Linear(out_planes[2], num_classes)
        
#         self.norm=norm

#     def _make_layer(self, out_planes, num_blocks, groups):
#         layers = []
#         for i in range(num_blocks):
#             stride = 2 if i == 0 else 1
#             cat_planes = self.in_planes if i == 0 else 0
#             layers.append(Bottleneck(self.in_planes, out_planes-cat_planes, stride=stride, groups=groups))
#             self.in_planes = out_planes
#         return nn.Sequential(*layers)


#     def extract_features(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = F.avg_pool2d(out, 4)
#         feature = out.view(out.size(0), -1)
        
#         normalized_feature= nn.functional.normalize(feature, p=2, dim=1)#feature normalize!!

#         return normalized_feature


#     def forward(self, x):
#         normalized_feature = self.extract_features(x)
                
#         logits = self.classifier(normalized_feature*self.norm)
        
#         return logits   
    
    
# import torch.nn as nn
# import torch.nn.functional as F

# __all__ = ["resnet18", "resnet18_fn"]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=False),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.classifier(feature)
        return out

    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)

        return feature


class ResNet_FN(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm=1):
        super(ResNet_FN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)
        
        self.norm=norm

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        
        normalized_feature= nn.functional.normalize(feature, p=2, dim=1)#feature normalize!!
        logits = self.classifier(normalized_feature*self.norm)
        
        return logits   

    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        
        normalized_feature= nn.functional.normalize(feature, p=2, dim=1)#feature normalize!!

        return normalized_feature

def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        
def resnet18_fn(num_classes=10, norm=1):
    return ResNet_FN(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, norm=norm)
    
    
