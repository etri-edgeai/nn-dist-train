import argparse
import logging
import os
import random
import sys
import copy

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))

from fedml_api.data_preprocessing.svhn.data_loader import load_partition_data_svhn, get_unlabeled_dataloader_SVHN
from fedml_api.data_preprocessing.cifar10.data_loader import load_partition_data_cifar10, get_unlabeled_dataloader_CIFAR10
from fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100, get_unlabeled_dataloader_CIFAR100
from fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from fedml_api.data_preprocessing.fed_cifar100.data_loader import load_partition_data_federated_cifar100
from fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare
from fedml_api.data_preprocessing.fed_shakespeare.data_loader import load_partition_data_federated_shakespeare
from fedml_api.data_preprocessing.stackoverflow_lr.data_loader import load_partition_data_federated_stackoverflow_lr
from fedml_api.data_preprocessing.stackoverflow_nwp.data_loader import load_partition_data_federated_stackoverflow_nwp
from fedml_api.data_preprocessing.ImageNet.data_loader import load_partition_data_ImageNet
from fedml_api.data_preprocessing.Landmarks.data_loader import load_partition_data_landmarks
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56, resnet8
from fedml_api.model.cv.cnn import CNN_DropOut
from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow

from fedml_api.data_preprocessing.MNIST.data_loader import load_partition_data_mnist
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.cv.resnet_gn import resnet18
from fedml_api.model.cv.resnet_wo_bn import resnet8_cifar as resnet8_no_bn
from fedml_api.model.cv.cnn import CNNCifar, CNN_OriginalFedAvg


from fedml_api.standalone.feddf.feddf_api import FeddfAPI
from fedml_api.standalone.feddf.fedcon_init_api import FeddfAPI as Fedcon_initAPI

from fedml_api.standalone.feddf.my_model_trainer_ensemble import MyModelTrainer as MyModelTrainerENS
from fedml_api.standalone.feddf.my_model_trainer_ensemble import MyModelTrainer_full_logits as MyModelTrainerENS_full
from fedml_api.standalone.feddf.my_model_trainer_ensemble import MyModelTrainer_fedmix as MyModelTrainerENS_Fedmix
from fedml_api.standalone.feddf.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from fedml_api.standalone.feddf.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from fedml_api.standalone.feddf.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
from fedml_api.standalone.feddf.my_model_trainer_classification_fedmix import MyModelTrainer as MyModelTrainerFedmix
from utils.utils import set_logger


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--model', type=str, default='resnet56', metavar='N',
                        help='neural network used in training')

    parser.add_argument('--dataset', type=str, default='cifar10', metavar='N',
                        help='dataset used for training')

    parser.add_argument('--data_dir', type=str, default='/home/osilab7/hdd/cifar',
                        help='data directory')

    parser.add_argument('--partition_method', type=str, default='hetero', metavar='N',
                        help='how to partition the dataset on local workers')

    parser.add_argument('--partition_alpha', type=float, default=0.5, metavar='PA',
                        help='partition alpha (default: 0.5)')

    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--client_optimizer', type=str, default='sgd',
                        help='SGD with momentum; adam')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1)')

    # parser.add_argument('--wd', help='weight decay parameter;', type=float, default=0.001)

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

    parser.add_argument('--seed', type=int, default=0,
                        help="Seed")

    parser.add_argument('--split_unequally', help='Split unequally?', default=False,
                        action='store_true')
    # Fedavg
    parser.add_argument('--uniform_aggregate', help='Uniform averaging in aggregation', default=False,
                        action='store_true')    
    
    parser.add_argument('--save_server', help='save server?',
                        action='store_true')
    
    # FedProx
    parser.add_argument('--lambda_fedprox', type=float, default=0.0, metavar='LR',
                        help='Fedprox lambda')
    
    
    # WANDB
    parser.add_argument('--pname', type=str, default='', metavar='N',
                        help='Project name for wandb ;con;hard')

    # For Multi augmentation

    parser.add_argument('--randaug', help='Use rand aug for labeled dataset (clients)',
                        action='store_true')

    parser.add_argument('--unlabeled_randaug', help='Use rand aug for unlabeled dataset (server)',
                        action='store_true')

    # For Ensemble distillation

    parser.add_argument('--unlabeled_data_dir', type=str, default='/home/osilab7/hdd/cifar', metavar='N',
                        help='Unlabeled dataset used for ensemble')

    parser.add_argument('--unlabeled_dataset', type=str, default='cifar100', metavar='N',
                        help='Unlabeled dataset used for ensemble')

    parser.add_argument('--unlabeled_batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')

    parser.add_argument('--server_steps', type=int, default=1e4, metavar='EP',
                        help='how many steps will be trained in the server')

    parser.add_argument('--server_patience_steps', type=int, default=1e3, metavar='EP',
                        help='how many steps will be trained in the server without increase in val acc')

    parser.add_argument('--server_lr', type=float, default=0.001, metavar='LR',
                        help='learning rate on server (default: 0.001)')

    parser.add_argument('--valid_ratio', type=float, default=0.0, metavar='LR',
                        help='Ratio of validation set')
    
    parser.add_argument('--train_ratio', type=float, default=1.0, metavar='LR',
                        help='Ratio of training set')

    parser.add_argument('--logit_type', type=str, default='average', metavar='LR',
                        help='Type of logit')
    
    # For fedmix

    parser.add_argument('--fedmix', help='Use fedmix?',
                        action='store_true')
    
    parser.add_argument('--fedmix_server', help='Use fedmix on Server?',
                        action='store_true')
    
    parser.add_argument('--lam', type=float, default=0.1, 
                        help="lambda fixed")
    
    parser.add_argument('--fedmix_wth_condense', help='Use condense data with averaged data',
                        action='store_true')
    
    parser.add_argument('--num_mixed_data_per_client', help="Condensing iterations",
                       type=int, default=0)
    
    
    
    # For Dataset Condensation
    
    parser.add_argument('--condense', help='Condensing?',
                        action='store_true')
    
    parser.add_argument('--image_per_class', help="Condense image per class",
                       type=int, default=0)
    
    parser.add_argument('--image_lr', type=float, default=0.1, help="lambda fixed")

    parser.add_argument('--outer_loops', help="Condensing iterations",
                       type=int, default=10)
    
    '''condense_init'''
    parser.add_argument('--condense_init', help='condensing at initializaiton only',
                        action='store_true') 
    
    parser.add_argument('--condense_reinit_model', help='Whether to reinit model while condensing',
                        action='store_true') 
    
    parser.add_argument('--init_outer_loops', help="Condensing iterations",
                       type=int, default=100)
  
    parser.add_argument('--coninit_load', help='Load condense data',
                        action='store_true')
    
    parser.add_argument('--coninit_load_dir', type=str, default='./condata',
                        help='data directory')

    parser.add_argument('--coninit_save', help='Save condense data',
                        action='store_true')

    parser.add_argument('--coninit_save_dir', type=str, default='./condata',
                        help='data directory')

    '''condense_mid'''
    parser.add_argument('--condense_mid', help='condensing in middle of round only',
                        action='store_true') 
    
    parser.add_argument('--condense_mid_round', help='When to start condensing',
                        type=int, default=50) 
    
    '''condense with public'''
    parser.add_argument('--train_public_condense_server', help='integrate public wth condense data',
                        action='store_true')    

  
    
    # Condense training

    
    parser.add_argument('--train_condense_server', help='train server with condense server',
                        action='store_true')
    
    parser.add_argument('--condense_server_steps', type=int, default=100, metavar='EP',
                        help='how many steps will be trained with condensed data in the server')
    
    parser.add_argument('--condense_patience_steps', type=int, default=100, metavar='EP',
                        help='how many patience steps will be trained with condensed data in the server')
    
    parser.add_argument('--condense_lr', type=float, default=0.001, 
                        help="lr condense training ")
    
    parser.add_argument('--condense_train_type', type=str, default='soft', metavar='N',
                        help='type of training method for condense training')

    parser.add_argument('--con_rand', help='train server with condense server',
                        action='store_true')
    
    parser.add_argument('--con_rand_neighbor', type=int, default=0, metavar='N',
                        help='number of neighbors to random condense')
    
    parser.add_argument('--condense_batch_size', type=int, default=6, metavar='N',
                        help='input batch size for training (default: 128)')
    
    parser.add_argument('--condense_optimizer', type=str, default='adam',
                        help='Adam')
    
    # Hard Sampling
    parser.add_argument('--hard_sample', type=str, default='', metavar='LR',
                        help='Type of hard sampling method')
    
    parser.add_argument('--hard_sample_ratio', type=float, default=0.0, metavar='LR',
                        help='Ratio of hard sample')
   
    # Test
    parser.add_argument('--train_condense_server_onebyone', help='train server with condense client one by one',
                        action='store_true')    
    

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
        
    # Channels for condensation
    args.channel = 3

    if dataset_name == "mnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        
        args.channel = 1
        
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, *valid_idxs = load_partition_data_mnist(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size,
                                             args.valid_ratio, split_equally=args.split_equally, randaug=args.randaug, condense=args.condense)    

    elif dataset_name == "femnist":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_federated_emnist(args.dataset, args.data_dir)
        args.client_num_in_total = client_num
        
    elif dataset_name == "svhn":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, *valid_idxs = load_partition_data_svhn(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size,
                                             args.valid_ratio, split_equally=args.split_equally, randaug=args.randaug, condense=args.condense)    
        

    else:
        if dataset_name == "cifar10":
            data_loader = load_partition_data_cifar10
        elif dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100
        elif dataset_name == "cinic10":
            data_loader = load_partition_data_cinic10
        else:
            raise ValueError("dataset {} has not been defined".format(dataset_name))

        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num, *valid_idxs = data_loader(args.dataset, args.data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total, args.batch_size,
                                             args.valid_ratio, split_equally=args.split_equally, randaug=args.randaug, condense=args.condense, train_ratio=args.train_ratio)    
        
        
    if centralized:
        train_data_local_num_dict = {
            0: sum(user_train_data_num for user_train_data_num in train_data_local_num_dict.values())}
        train_data_local_dict = {
            0: [batch for cid in sorted(train_data_local_dict.keys()) for batch in train_data_local_dict[cid]]}
        test_data_local_dict = {
            0: [batch for cid in sorted(test_data_local_dict.keys()) for batch in test_data_local_dict[cid]]}
        args.client_num_in_total = 1

    if full_batch:
        train_data_global = combine_batches(train_data_global)
        test_data_global = combine_batches(test_data_global)
        train_data_local_dict = {cid: combine_batches(train_data_local_dict[cid]) for cid in
                                 train_data_local_dict.keys()}
        test_data_local_dict = {cid: combine_batches(test_data_local_dict[cid]) for cid in test_data_local_dict.keys()}
        args.batch_size = args_batch_size

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]

    if valid_idxs:
        # If validation exist
        valid_data_global = valid_idxs[0]
        dataset.append(valid_data_global)
    
    if args.condense :
        
        if len(valid_idxs) == 2 :
            data_local_noaug = valid_idxs[1]
        else : 
            data_local_noaug = valid_idxs[0]
        dataset.append(data_local_noaug)
        
    
    ## class_num
    args.class_num = class_num
    
    return dataset

def load_unlabeled_data(args, dataset_name):

    logging.info("{} as unlabeled dataset".format(dataset_name))
    # Unlabeled data dir
    if args.unlabeled_data_dir == "":
        args.unlabeled_data_dir = args.data_dir

    
    if dataset_name == "cifar10":
        train_data_num, test_data_num, train_dl, test_dl = get_unlabeled_dataloader_CIFAR10(args.unlabeled_data_dir,
                                                                                            args.unlabeled_batch_size,
                                                                                            args.unlabeled_batch_size,
                                                                                            randaug=args.unlabeled_randaug)
    elif dataset_name == "cifar100":
        train_data_num, test_data_num, train_dl, test_dl = get_unlabeled_dataloader_CIFAR100(args.unlabeled_data_dir,
                                                                                            args.unlabeled_batch_size,
                                                                                            args.unlabeled_batch_size,
                                                                                            randaug=args.unlabeled_randaug)
    elif dataset_name == "svhn":
        train_data_num, test_data_num, train_dl, test_dl = get_unlabeled_dataloader_SVHN(args.unlabeled_data_dir,
                                                                                             args.unlabeled_batch_size,
                                                                                             args.unlabeled_batch_size,
                                                                                         randaug=args.unlabeled_randaug)

    else :
        raise ValueError("{} not defined".format(dataset_name))

    dataset = [train_data_num, test_data_num, train_dl, test_dl]
    
    return dataset


def combine_batches(batches):
    full_x = torch.from_numpy(np.asarray([])).float()
    full_y = torch.from_numpy(np.asarray([])).long()
    for (batched_x, batched_y) in batches:
        full_x = torch.cat((full_x, batched_x), 0)
        full_y = torch.cat((full_y, batched_y), 0)
    return [(full_x, full_y)]


def create_model(args, model_name, output_dim):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "resnet8":
        model = resnet8(class_num=output_dim)
    elif model_name == "resnet8_no_bn" and "cifar" in args.dataset:
        model = resnet8_no_bn(dataset=args.dataset)
    elif model_name == "resnet8_no_bn" and "svhn" in args.dataset:
        model = resnet8_no_bn(dataset=args.dataset)
    elif model_name == "cnn" and "cifar" in args.dataset:
        model = CNNCifar(class_num=output_dim)
    elif model_name == "cnn" and "mnist" in args.dataset:
        model = CNN_OriginalFedAvg(class_num=output_dim)
    else :
        raise ValueError("No model name {} , {}".format(model_name, args.dataset))

    return model


def custom_model_trainer(args, model, ensemble=None, logit_type='average'):
    

    
    if ensemble == True:
        
        if args.fedmix_server :
            return MyModelTrainerENS_Fedmix(model)
        if logit_type == 'average':
            return MyModelTrainerENS(model)
        elif logit_type == 'full' :
            return MyModelTrainerENS_full(model)
    elif args.fedmix :
            return MyModelTrainerFedmix(model)
    
    if args.dataset == "stackoverflow_lr":
        return MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        return MyModelTrainerNWP(model)
    else: # default model trainer is for classification problem
        return MyModelTrainerCLS(model)


def get_proj_name(pname):
    if pname == "con":
        display_name = "FedCon" + \
             "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
    "-cps" + str(args.condense_patience_steps) + "-css" + str(args.condense_server_steps) + \
       "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-0519"
        
    elif pname == "con-init":
        display_name = "FedCon-init" + \
             "-localepoch_" + str(args.epochs) + "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
       "-coninit_" + str(args.condense_init) + "-initol_" + str(args.init_outer_loops) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
    "-cps" + str(args.condense_patience_steps) + "-css" + str(args.condense_server_steps) + "-s" + str(args.seed) + \
       "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-0519"
        
    elif pname == "con-reinit":
        display_name = "FedCon-reinit" + \
             "-localepoch_" + str(args.epochs) + "-client_num" + str(args.client_num_in_total)  +\
        "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
       "-coninit_" + str(args.condense_init) + "-initol_" + str(args.init_outer_loops) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
    "-cps" + str(args.condense_patience_steps) + "-css" + str(args.condense_server_steps) + "-s" + str(args.seed) + "-trainratio" + str(args.train_ratio) + \
       "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-0519"
        
    elif pname == "con-neighbor": # condensing neighbor
        display_name = "FedCon-reinit-df" + \
             "-localepoch_" + str(args.epochs) + "-client_num" + str(args.client_num_in_total)  +\
        "-alpha" + str(args.partition_alpha) + "-neighbor" + str(args.con_rand_neighbor) + \
       "-coninit_" + str(args.condense_init) + "-initol_" + str(args.init_outer_loops) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
    "-cps" + str(args.condense_patience_steps) + "-css" + str(args.condense_server_steps) + "-s" + str(args.seed) +  \
       "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-0606"
        
    elif pname == "con-df": # Condense data + public data
        display_name = "FedCon-reinit-df" + \
             "-localepoch_" + str(args.epochs) + "-client_num" + str(args.client_num_in_total)  +\
        "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
        "-bs" + str(args.unlabeled_batch_size) + "-conbs" + str(args.condense_batch_size) + \
       "-coninit_" + str(args.condense_init) + "-initol_" + str(args.init_outer_loops) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
    "-cps" + str(args.condense_patience_steps) + "-css" + str(args.condense_server_steps) + "-s" + str(args.seed) + "-trainratio" + str(args.train_ratio) + \
       "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-0606"
        
    elif pname == "con-mid":
        display_name = "FedCon-mid" + \
             "-localepoch_" + str(args.epochs) + "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
       "-coninit_" + str(args.condense_init) + "-initol_" + str(args.init_outer_loops) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
    "-cps" + str(args.condense_patience_steps) + "-css" + str(args.condense_server_steps) +  "-s" + str(args.seed) + \
       "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-0519"

    elif pname == "con-reinit-onebyone":
        display_name = "FedCon-re-onebyone" + \
             "-localepoch_" + str(args.epochs) + "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
       "-coninit_" + str(args.condense_init) + "-initol_" + str(args.init_outer_loops) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
    "-cps" + str(args.condense_patience_steps) + "-css" + str(args.condense_server_steps) + \
       "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-0519"
        
    elif pname == "con-data-hp":
        display_name = "FedCon-data-hp" + \
             "-localepoch_" + str(args.epochs) + "-alpha" + str(args.partition_alpha) + \
        "-iol" + str(args.init_outer_loops) + "-imglr" + str(args.image_lr) + "ipc" + str(args.image_per_class) + \
         "-seed_" + str(args.seed) + "-cn_" + str(args.client_num_in_total) + "-model_" + str(args.model) 
        project_name = "fedcon-data-hp"
        
    elif pname == "con-init-hp":
        display_name = "FedCon-init-hp" + \
             "-localepoch_" + str(args.epochs) + "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + "-conbatchsize_" + str(args.condense_batch_size) + \
        "-conlr_"  + str(args.condense_lr) + "-conopt_" + str(args.condense_optimizer) + \
       "-coninit_" + str(args.condense_init) + "-initol_" + str(args.init_outer_loops) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
    "-cps" + str(args.condense_patience_steps) + "-css" + str(args.condense_server_steps) + \
       "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-hp"
    
    elif pname == "hard":
        display_name = "Feddf-Hard" + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-hardtype" + str(args.hard_sample) + "-hardratio" + str(args.hard_sample_ratio) + \
   "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-0519"

    elif pname == "feddf":
        display_name = "Feddf" + "-localepoch_" + str(args.epochs)  + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-localepoch_" + str(args.epochs) + \
   "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-0519"
    
    elif pname == "feddf-svhn":
        display_name = "Feddf-svhn" + "-localepoch_" + str(args.epochs)  + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-localepoch_" + str(args.epochs) + \
   "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model)
        project_name = "fedcon-0519"
    
    elif pname == "fedmix":
        display_name = "Fedmix" + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-localepoch_" + str(args.epochs) + "-unlabeldata_" + str(args.unlabeled_dataset) + \
   "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-fedmixServer_" + \
        str(args.fedmix_server) + "-model_" + str(args.model)
        project_name = "fedmix-0603"
        
    elif pname == "fedmix_reinit":
        display_name = "Fedmix_reinit" + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-localepoch_" + str(args.epochs) + "-unlabeldata_" + str(args.unlabeled_dataset) + \
        "-coninit_" + str(args.condense_init) + "-initol_" + str(args.init_outer_loops) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
   "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-fedmixServer_" + \
        str(args.fedmix_server) + "-model_" + str(args.model)
        project_name = "fedmix-0603"
    
    elif pname == "avg":
        display_name = "Fedavg" + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-localepoch_" + str(args.epochs) + "-model_" + str(args.model)  + "-unlabeldata_" + str(args.unlabeled_dataset) + \
    "-fedmix_" + str(args.fedmix)  
        project_name = "fedavg-0420"
        
    elif pname == "avg-svhn":
        display_name = "Fedavg-svhn" + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-localepoch_" + str(args.epochs) + "-model_" + str(args.model)  + "-unlabeldata_" + str(args.unlabeled_dataset) + \
    "-fedmix_" + str(args.fedmix)  
        project_name = "fedavg-0420"
    
    elif pname == "prox":
        display_name = "FedProx" + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-localepoch_" + str(args.epochs) + "-model_" + str(args.model)  + "-unlabeldata_" + str(args.unlabeled_dataset) + \
    "-fedmix_" + str(args.fedmix)  + "-prox_" + str(args.lambda_fedprox)
        project_name = "fedavg-0420"
    
    elif pname == "prox-df":
        display_name = "Feddf-prox" + "-localepoch_" + str(args.epochs)  + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-localepoch_" + str(args.epochs) + \
   "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model) + "-prox_" + str(args.lambda_fedprox)
        project_name = "fedcon-0519"
    
    elif pname == "prox-con":
        display_name = "FedCon-prox" + "-localepoch_" + str(args.epochs)  + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-localepoch_" + str(args.epochs) + \
        "-coninit_" + str(args.condense_init) + "-initol_" + str(args.init_outer_loops) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
    "-cps" + str(args.condense_patience_steps) + "-css" + str(args.condense_server_steps) + "-s" + str(args.seed) + \
   "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model) + \
        "-prox_" + str(args.lambda_fedprox)
        project_name = "fedcon-0519"
    
    elif pname == "prox-con-re":
        display_name = "FedCon-prox-re" + "-localepoch_" + str(args.epochs)  + \
         "-alpha" + str(args.partition_alpha) + "-ssteps_" +  str(args.server_steps) + \
    "-localepoch_" + str(args.epochs) + \
        "-coninit_" + str(args.condense_init) + "-initol_" + str(args.init_outer_loops) + \
        "-contype_" + str(args.condense_train_type) + "-ol" + str(args.outer_loops) + \
    "-cps" + str(args.condense_patience_steps) + "-css" + str(args.condense_server_steps) + "-s" + str(args.seed) + \
   "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-model_" + str(args.model) + \
        "-prox_" + str(args.lambda_fedprox)
        project_name = "fedcon-0519"
    
    
    elif pname == "test":
        display_name = "Pre condensing data" + \
         "-alpha" + str(args.partition_alpha) + "-dset_" + str(args.dataset) + \
   "-unlabel" + str(args.unlabeled_dataset) + "-fedmix_" + str(args.fedmix) + "-conrand_" + str(args.con_rand)
        project_name = "testing"
        
    else :
        raise ValueError("Project name not defined")
        
    return display_name, project_name

if __name__ == "__main__":

    parser = add_args(argparse.ArgumentParser(description='FedDF'))
    args = parser.parse_args()
    display_name, project_name = get_proj_name(args.pname)
    save_dir = os.path.join("./logs", display_name)
    os.makedirs(save_dir, exist_ok=True)
        
    wandb.init(
        project=project_name,
        name=display_name, 
        config=args,
        dir=save_dir
    )
    

    
    # Set Logger
    wandb_save_dir = wandb.run.dir
    set_logger(os.path.join(wandb_save_dir, 'log.log'))
    args.wandb_save_dir = wandb_save_dir
    logging.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logging.info(device)


    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # load data and unlabeled data
    args.split_equally = not args.split_unequally
    dataset = load_data(args, args.dataset)
    unlabeled_dataset = load_unlabeled_data(args, args.unlabeled_dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    server_model_trainer = custom_model_trainer(args, model, ensemble=True, logit_type=args.logit_type)
    client_model_trainer = custom_model_trainer(args, copy.deepcopy(model))
    model_trainer = [server_model_trainer, client_model_trainer]
    
    logging.info(model)
    

    feddfAPI = FeddfAPI(dataset, unlabeled_dataset, device, args, model_trainer)
    feddfAPI.train()
