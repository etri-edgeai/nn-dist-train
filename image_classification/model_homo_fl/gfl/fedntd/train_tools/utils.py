from .models import *

__all__ = ["create_models"]

MODELS = {
    "motivcnn": fedavgnet.MotivCNN,
    "simplecnn": fedavgnet.SimpleCNN,
    "fedavg_mnist": fedavgnet.FedAvgNetMNIST,
    "fedavg_cifar": fedavgnet.FedAvgNetCIFAR,
    "fedavg_cifar_add": fedavgnet.FedAvgNetCIFARAdd,
    "motivcnn_add": fedavgnet.MotivCNNAdd,
    "fedavg_tiny": fedavgnet.FedAvgNetTiny,
    "vgg11": vgg.vgg11,
    "res10": resnet.resnet10,
    "res34": resnet.resnet34,
    "mobile": mobilenet,
}

NUM_CLASSES = {
    "femnist": 62,
    "mnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "fed_cifar100": 100,
    "cinic10": 10,
    "tinyimagenet": 200,
}


def create_models(model_name, dataset_name):
    """Create a network model"""

    num_classes = NUM_CLASSES[dataset_name]
    model = MODELS[model_name](num_classes=num_classes)

    return model
