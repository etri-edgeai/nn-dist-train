import os
import torch

from datasetter.data_loader import DataSetter

IP_list = ['192.168.0.10', '192.168.0.11', '192.168.0.15']

# datasets
datasetter = DataSetter(root='./data', dataset='cifar10')
datasets = datasetter.data_distributer(
    alg='fedma',
    max_class_num=10,
    n_clients=3,
)

if not os.path.isdir('./checkpoint'):
    os.makedirs('./checkpoint')

testset = datasets['test']
# testloader = DataLoader(testset, batch_size=64)

torch.save(testset, './data/testset.pth')

for i in range(len(IP_list)):
    torch.save(datasets['local'][i], './data/trainset.pth')
    
    os.system('scp ./data/trainset.pth osilab@%s:~/nn-dist-train/Jetson/data' % IP_list[i])
