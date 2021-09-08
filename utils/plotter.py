import os
import torch
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['clients_data_num_plotter', 'test_acc_plotter']


def clients_data_num_plotter(args, clients_data_num):
    num_clients = len(clients_data_num)
    num_classes = len(clients_data_num[0])
    
    classwise_data_stat = {}
    for cls in range(num_classes):
        classwise_data_stat[cls] = [clients_data_num[c][cls] for c in range(num_clients)]
        
    x = range(num_clients)
    labels = ['C{}'.format(number) for number in range(num_clients)]
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    tmp = np.zeros(num_clients)

    f = plt.figure(figsize=(20,10))
    for i in range(num_classes):
        plt.bar(x, classwise_data_stat[i], bottom = tmp, label = classes[i])
        bottom = np.add(classwise_data_stat[i], tmp)
        tmp += classwise_data_stat[i]
    plt.rc('legend', fontsize=18)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    ax = plt.subplot()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if 'class' in args.img_path:
        num = int((50000 - args.valid_size * num_classes) / num_clients)
        ax.set_ylim(0, num * 1.05)
        
    plt.title('Dirichilet data distribution with non-iid {}'.format(args.non_iid))
    plt.xlabel('Clients')
    plt.ylabel('number of data')
    plt.legend(loc='upper center', ncol=10, bbox_to_anchor= (0.5,-0.05))
    
    f.savefig(args.img_path + '/clients_data_num.png', bbox_inches = 'tight')
    plt.show()
    plt.close()


def test_acc_plotter(args, test_acc):    
    x = range(args.num_rounds)

    f = plt.figure(figsize=(20,10))
    plt.plot(x, test_acc)
    
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    ax = plt.subplot()
    ax.set_xticks(x)
    
    f.savefig(args.img_path + '/test_acc.png', bbox_inches='tight')
    plt.close()