import argparse

__all__ = ['parse_args']
        
        
class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k,v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)
        

def parse_args():
    parser = argparse.ArgumentParser()

    ############ experimental settings ###############
    parser.add_argument('--device', 
                    help='gpu device;', 
                    type=str,
                    default='cuda:0')
    parser.add_argument('--seed',
                    help='seed for experiment',
                    type=int,
                    default=0)
    
    #################### dataset ####################
    # basic settings
    parser.add_argument('--dataset',
                    help='name of dataset;',
                    type=str,
                    required=True,
                    choices=['dirichlet-mnist', 'dirichlet-cifar10', 'dirichlet-cifar100', 'dirichlet-fmnist', 'emnist', 'fed-cifar100', 'synthetic', 'landmark-g23k', 'landmark-g160k'])
    parser.add_argument('--data-dir', 
                    help='dir for dataset;',
                    type=str,
                    default='./data')
    parser.add_argument('--batch-size',
                    help='batch size of local data on each client;',
                    type=int,
                    default=128)
    parser.add_argument('--pin-memory', 
                    help='argument of pin memory on DataLoader;',
                    action='store_true')
    parser.add_argument('--num-workers', 
                    help='argument of num workers on DataLoader;',
                    type=int,
                    default=4)
    parser.add_argument('--num-clients', 
                    help='number of clients;',
                    type=int,
                    default=20)
    
    # non-iidness
    parser.add_argument('--dist-mode', 
                    help='which criterion to use on distributing the dataset;',
                    type=str,
                    default='class')
    parser.add_argument('--non-iid', 
                    help='dirichlet parameter to control non-iidness of dataset;',
                    type=float,
                    default=10.0)
    
    ##################### model #####################
    parser.add_argument('--model',
                    help='name of model;',
                    type=str,
                    required=True,
                    choices=['lenet', 'lenetcontainer', 'vgg11', 'vgg11-bn', 'vgg13', 'vgg13-bn', 'vgg16', 'vgg16-bn', 'vgg19', 'vgg19-bn', 'resnet8'])
    # example : --model-kwargs num_classes=10
    parser.add_argument('--model-kwargs',
                        dest='model_kwargs',
                        action=StoreDictKeyPair,
                        metavar="KEY1=VAL1,KEY2=VAL2...")
    
    ################## server_opt ###################
    parser.add_argument('--algorithm',
                    help='which algorithm to select clients;',
                    type=str,
                    default='fedavg')
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=100)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=str,
                    default='20')
    parser.add_argument('--global-momentum',
                    help='whether to use momentum in server optimizers;',
                    type=float,
                    default=0.0)
    
    ################## client_opt ###################
    parser.add_argument('--num-epochs',
                    help='number of rounds to local update;',
                    type=int,
                    default=1)
    # criterion
    parser.add_argument('--local-criterion',
                    help='criterion to use in local training;',
                    type=str,
                    default='ce')
    
    # optimizer
    parser.add_argument('--local-optimizer',
                    help='optimizer to use in local training;',
                    type=str,
                    default='sgd')
    parser.add_argument('--local-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=0.1)
    parser.add_argument('--wd',
                    help='weight decay lambda hyperparameter in local optimizers;',
                    type=float,
                    default=1e-4)
    parser.add_argument('--mu',
                    help='fedprox mu hyperparameter in local optimizers;',
                    type=float,
                    default=0.0)
    parser.add_argument('--nesterov',
                    help='nesterov switch for local momentum',
                    action='store_true')
    parser.add_argument('--local-momentum',
                    help='whether to use momentum in local optimizers;',
                    type=float,
                    default=0.9)
    parser.add_argument('--lr-decay',
                    help = 'learning rate decay',
                    type = float,
                    default = 0.1)
    parser.add_argument('--milestones',
                    help= 'milestones for step scheduler',
                    type = str,
                    default = '50,75')
    parser.add_argument('--sch-type',
                       help = 'scheduler for local client learning rate',
                       type=str,
                       default='uniform')
    
    return parser.parse_args()