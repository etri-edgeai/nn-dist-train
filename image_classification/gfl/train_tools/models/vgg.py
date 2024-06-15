import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomHardtanh(nn.Module):
    def __init__(self, initial_param=1):
        super(CustomHardtanh, self).__init__()
        self.param = nn.Parameter(torch.tensor(float(initial_param), requires_grad=True))

    def forward(self, x):
        abs_param = torch.abs(self.param)
        return F.relu(x - abs_param) + abs_param - F.relu(x + abs_param)

class VGG(nn.Module):
    def __init__(
        self,
        features,
        use_dropout=False,
        fc_dim=512,
        num_classes=10,
        img_size=3 * 32 * 32,
        use_bias=True,
    ):
        super(VGG, self).__init__()

        self.features = features

        if img_size == 1 * 28 * 28:
            fc_dimin = 512
        else:
            fc_dimin = 512 * int(img_size / (3 * 32 * 32))

        if use_dropout:
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(fc_dimin, fc_dim),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(fc_dimin, fc_dim),
                nn.ReLU(True),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(True),
            )

        # for tiny-imagenet case
        self.classifier = nn.Linear(fc_dim, num_classes, bias=use_bias)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        x = self.classifier(features)
        return x
    
    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
    
        return features
    
    
    
    
class VGG_Sphere(nn.Module):
    def __init__(
        self,
        features,
        use_dropout=False,
        fc_dim=512,
        num_classes=10,
        img_size=3 * 32 * 32,
        use_bias=True, norm=1
    ):
        super(VGG_Sphere, self).__init__()

        self.features = features

        if img_size == 1 * 28 * 28:
            fc_dimin = 512
        else:
            fc_dimin = 512 * int(img_size / (3 * 32 * 32))

        if use_dropout:
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(fc_dimin, fc_dim),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(fc_dimin, fc_dim),
                nn.ReLU(True),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(True),
            )

        # for tiny-imagenet case
        self.classifier = nn.Linear(fc_dim, num_classes, bias=False)
        self.norm = norm

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        x = self.classifier(normalized_feature*self.norm)
        
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        return normalized_feature
    
    
class VGG_ETF(nn.Module):
    def __init__(
        self,
        features,
        use_dropout=False,
        fc_dim=512,
        num_classes=10,
        img_size=3 * 32 * 32,
        use_bias=True, norm=1
    ):
        super(VGG_ETF, self).__init__()

        self.features = features

        if img_size == 1 * 28 * 28:
            fc_dimin = 512
        else:
            fc_dimin = 512 * int(img_size / (3 * 32 * 32))

        if use_dropout:
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(fc_dimin, fc_dim),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(fc_dimin, fc_dim),
                nn.ReLU(True),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(True),
            )

        # for tiny-imagenet case
        self.projection= nn.Linear(fc_dim, num_classes, bias=False)
        self.classifier = nn.Linear(num_classes, num_classes, bias=False)
        self.norm = nn.Parameter(torch.tensor([norm], dtype=torch.float))

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        features= self.projection(features)
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        x = self.classifier(normalized_feature*self.norm)
        
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        features= self.projection(features)
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        return normalized_feature
    
    
    
    
class VGG_NCP(nn.Module):
    def __init__(
        self,
        features,
        use_dropout=False,
        fc_dim=512,
        num_classes=10,
        img_size=3 * 32 * 32,
        use_bias=True, norm=1
    ):
        super(VGG_NCP, self).__init__()

        self.features = features
        self.norm=norm
        self.custom_hardtanh = CustomHardtanh()

        if img_size == 1 * 28 * 28:
            fc_dimin = 512
        else:
            fc_dimin = 512 * int(img_size / (3 * 32 * 32))

        if use_dropout:
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(fc_dimin, fc_dim),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(fc_dim, fc_dim),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(fc_dimin, fc_dim),
                nn.ReLU(True),
                nn.Linear(fc_dim, fc_dim),
            )

        # for tiny-imagenet case
        self.classifier = nn.Linear(fc_dim, num_classes, bias=False)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.custom_hardtanh(self.fc(x))
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        x = self.classifier(normalized_feature*self.norm)
        
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.custom_hardtanh(self.fc(x))
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        return normalized_feature
    
    
class VGG_DR(nn.Module):
    def __init__(
        self,
        features,
        use_dropout=False,
        fc_dim=512,
        num_classes=10,
        img_size=3 * 32 * 32,
        use_bias=True, norm=1
    ):
        super(VGG_DR, self).__init__()

        self.features = features

        if img_size == 1 * 28 * 28:
            fc_dimin = 512
        else:
            fc_dimin = 512 * int(img_size / (3 * 32 * 32))

        if use_dropout:
            self.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(fc_dimin, fc_dim),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(fc_dimin, fc_dim),
                nn.ReLU(True),
                nn.Linear(fc_dim, fc_dim),
                nn.ReLU(True),
            )

        # for tiny-imagenet case
        self.classifier = nn.Linear(fc_dim, num_classes, bias=False)
        self.norm = norm

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        x = self.classifier(normalized_feature*self.norm)
        
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        return normalized_feature
    
    def extract_original_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        return features
    
    
    
    
def make_layers(cfg, in_channels=3, batch_norm=False, img_size=3 * 32 * 32):
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    if img_size == 1 * 28 * 28:
        del layers[-1]

    return nn.Sequential(*layers)


cfg = {"vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]}


def vgg11(num_classes=10):
    return VGG(
        make_layers(cfg["vgg11"], in_channels=3, img_size=3 * 32 * 32),
        num_classes=num_classes
    )

def vgg11_Sphere(num_classes=10):
    return VGG_Sphere(
        make_layers(cfg["vgg11"], in_channels=3, img_size=3 * 32 * 32),
        num_classes=num_classes
    )

def vgg11_ETF(num_classes=10):
    return VGG_ETF(
        make_layers(cfg["vgg11"], in_channels=3, img_size=3 * 32 * 32),
        num_classes=num_classes
    )

def vgg11_NCP(num_classes=10):
    return VGG_NCP(
        make_layers(cfg["vgg11"], in_channels=3, img_size=3 * 32 * 32),
        num_classes=num_classes
    )

    
def vgg11_DR(num_classes=10):
    return VGG_DR(
        make_layers(cfg["vgg11"], in_channels=3, img_size=3 * 32 * 32),
        num_classes=num_classes
    )

