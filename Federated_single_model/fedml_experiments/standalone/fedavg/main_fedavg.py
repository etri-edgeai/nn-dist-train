import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

import wandb  

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10 
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100 

from fedml_api.model.cv.resnet import ResNet  
from fedml_api.model.cv.cnn import *  




from fedml_api.model.cv.vgg import * 


from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI 
from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS 



def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet', metavar='N',
                        help='neural network used in training')
    
    parser.add_argument('--depth', type=int, default=20,
                    help='model depth number')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='./../../../data/cifar10',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')
    
    parser.add_argument('--shard_ratio', type=float, default=0.2, metavar='PA',
                        help='heterogeneity of shard (default: 0.2)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')

    parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

    parser.add_argument('--epochs', type=int, default=5, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--client_num_in_total', type=int, default=10, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=10, metavar='NN',
                        help='number of workers')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--frequency_of_the_test', type=int, default=5,
                        help='the frequency of the algorithms')

    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')

    parser.add_argument('--ci', type=int, default=0,
                        help='CI')
    
    parser.add_argument('--mu', type=float, default=0,
                        help='prox parameter')
    
    parser.add_argument('--decaylr', type=float, default=1,
                        help='decay factor')
    
    parser.add_argument('--seed', type=int, default=0,
                        help='seed number')
    
    

    
    return parser


def load_data(args, dataset_name):
    # check if the centralized training is enabled
    centralized = True if args.client_num_in_total == 1 else False

    # check if the full-batch training is enabled
    args_batch_size = args.batch_size
    if args.batch_size <= 0:
        full_batch = True
        args.batch_size = 128  # temporary batch size
    else:
        full_batch = False

    else:
        if dataset_name == "cifar10": #########
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100": ###########
            data_loader = load_partition_data_cifar100
        else:
            data_loader = load_partition_data_cifar10
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size, args.shard_ratio)

    if centralized:#WORKER 1게로 통합!!
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:#BATCH SIZE를 모든 CLIENT에 대해서 FULL BATCH(CLIENT별 가지고 있는 전체 DATA)로 잡는다
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]      
    return dataset



def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "vgg11" :
        model = VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']), num_classes=output_dim)
    elif model_name == "resnet":
        model = ResNet(depth=args.depth, num_classes=output_dim)
    elif model_name == "cnn":
        model = CNN_DropOut_cifar(num_classes=output_dim).to(device) 

        
    return model



if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="InitCheck", entity="etri_2022",
        name="FedAVG-r" + str(args.comm_round)+"-m"+str(args.model)+str(args.depth) + "-e" + str(args.epochs) + "-lr" + str(args.lr)+ "-t" + str(args.client_num_in_total)+ "-d" + str(args.dataset)+ "-kind" + str(args.partition_method)+ "-alpha" + str(args.partition_alpha)+"-prox" + str(args.mu)+"-decay factor" + str(args.decaylr),
        config=args
    )


    
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    


    dataset = load_data(args, args.dataset)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])

    model_trainer = MyModelTrainerCLS (model)
    logging.info(model)

    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer)
    fedavgAPI.train()
