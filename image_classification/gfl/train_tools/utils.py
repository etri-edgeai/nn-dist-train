from .models import *

__all__ = ["create_models"]

MODELS = {
    "mobile": mobilenet.MobileNetCifar,
    "mobile_sphere": mobilenet.MobileNetCifar_Sphere,
    "mobile_etf": mobilenet.MobileNetCifar_ETF,
    "mobile_ncp": mobilenet.MobileNetCifar_NCP,
    "mobile_dr": mobilenet.MobileNetCifar_DR,
    "mobile_fn": mobilenet.MobileNetCifar_Sphere,
    
    
    "tiny_mobile": mobilenet.MobileNettiny,
    "tiny_mobile_sphere": mobilenet.MobileNettiny_Sphere,
    "tiny_mobile_etf": mobilenet.MobileNettiny_ETF,
    "tiny_mobile_ncp": mobilenet.MobileNettiny_NCP,
    "tiny_mobile_dr": mobilenet.MobileNettiny_DR,
    "tiny_mobile_fn": mobilenet.MobileNettiny_Sphere,
    
    
    "imagenet_mobile": mobilenet.MobileNetImageNet,
    "imagenet_mobile_sphere": mobilenet.MobileNetImageNet_Sphere,
    "imagenet_mobile_etf": mobilenet.MobileNetImageNet_ETF,
    "imagenet_mobile_ncp": mobilenet.MobileNetImageNet_NCP,
    "imagenet_mobile_dr": mobilenet.MobileNetImageNet_DR,
    "imagenet_mobile_fn": mobilenet.MobileNetImageNet_Sphere,
    
    
    "imagenet_res18": resnet.Res18ImageNet,
    "imagenet_res18_sphere": resnet.Res18ImageNet_Sphere,
    "imagenet_res18_etf": resnet.Res18ImageNet_ETF,
    "imagenet_res18_ncp": resnet.Res18ImageNet_NCP,
    "imagenet_res18_dr": resnet.Res18ImageNet_DR,
    "imagenet_res18_fn": resnet.Res18ImageNet_Sphere,    
    
    
    "vgg": vgg.vgg11,
    "vgg_sphere": vgg.vgg11_Sphere,
    "vgg_etf": vgg.vgg11_ETF,
    "vgg_ncp": vgg.vgg11_NCP,
    "vgg_dr": vgg.vgg11_DR,  
    "vgg_fn": vgg.vgg11_Sphere
    
}

NUM_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 200,
    "imagenet": 100,    
}


def create_models(model_name, dataset_name, **kwargs):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]
    model = MODELS[model_name](num_classes=num_classes)

    return model
