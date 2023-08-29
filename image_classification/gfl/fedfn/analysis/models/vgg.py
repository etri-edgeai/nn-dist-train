import math
import torch.nn as nn


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


class VGG_FN(nn.Module):
    def __init__(
        self,
        features,
        use_dropout=False,
        fc_dim=512,
        num_classes=10,
        img_size=3 * 32 * 32,
        use_bias=True, norm=1
    ):
        super(VGG_FN, self).__init__()

        self.features = features
        self.norm=norm

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
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        x = self.classifier(normalized_feature*self.norm)
        
        return x

    def extract_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        
        normalized_feature= nn.functional.normalize(features, p=2, dim=1)#feature normalize!!

        return normalized_feature
    
        
        
        
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

def vgg11_fn(num_classes=10, norm=1):
    return VGG_FN(
        make_layers(cfg["vgg11"], in_channels=3, img_size=3 * 32 * 32),
        num_classes=num_classes, norm=norm
    )