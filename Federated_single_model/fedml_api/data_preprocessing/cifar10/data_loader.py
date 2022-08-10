import logging

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from .datasets import CIFAR10_truncated

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# generate the non-IID distribution for all methods
def read_data_distribution(filename='./data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'):#안봐도 될 것 같음
    distribution = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0]:
                tmp = x.split(':')
                if '{' == tmp[1].strip():
                    first_level_key = int(tmp[0])
                    distribution[first_level_key] = {}
                else:
                    second_level_key = int(tmp[0])
                    distribution[first_level_key][second_level_key] = int(tmp[1].strip().replace(',', ''))
    return distribution


def read_net_dataidx_map(filename='./data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'):#안봐도 될 것 같음
    net_dataidx_map = {}
    with open(filename, 'r') as data:
        for x in data.readlines():
            if '{' != x[0] and '}' != x[0] and ']' != x[0]:
                tmp = x.split(':')
                if '[' == tmp[-1].strip():
                    key = int(tmp[0])
                    net_dataidx_map[key] = []
                else:
                    tmp_array = x.split(',')
                    net_dataidx_map[key] = [int(i.strip()) for i in tmp_array]
    return net_dataidx_map


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}#각 client가 어떤 label을 몇개씩 가지고 있는지 통계량 기재!!

    for net_i, dataidx in net_dataidx_map.items():#net_i: key(client index), data_idx: value(client가 가지고 있는 data index)
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)#전체 train data 중에 net_i번째 client가 가지고 있는 data가 어떤 label을 가지고 있는지의 정보가 unq, unq의 각 element가 몇개 들어있는지 기재하는게 unq_count이다!!
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}#tmp에는 unq가 key unq_count가 value가 되게 기재!!
        net_cls_counts[net_i] = tmp
    logging.debug('Data statistics: %s' % str(net_cls_counts))
    return net_cls_counts #각 client가 어떤 label을 몇개씩 가지고 있는지 통계량 기재!!


class Cutout(object):#_data_transforms_cifar10()에 적용됨!!
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_transform.transforms.append(Cutout(16))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    return train_transform, valid_transform


def load_cifar10_data(datadir):#partition_data에 쓰임!!
    train_transform, test_transform = _data_transforms_cifar10()

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=train_transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=test_transform)
    #위 두 개는 class이다. 
    #truncated method를 안먹였기에 truncated는 안된다.
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)#단순 data set만 업로드!! augmentation은 안먹인 것!!


def partition_data(dataset, datadir, partition, n_nets, alpha, ratio_shard):
    logging.info("*********partition data***************")
    X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    n_train = X_train.shape[0]
    # n_test = X_test.shape[0]
    class_num = len(np.unique(y_train))
    if partition == "homo":
        total_num = n_train
        idxs = np.random.permutation(total_num)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}
        
        
    elif partition == "shard": #각 client는 최대 class_num*ratio_shard개의 label을 가질 수 있는 상황(Cifar 10기준 20퍼센트 비율!!)
        num_shards= int(class_num*ratio_shard)*n_nets #ratio_shard=0.2, n_nets=100으로 짜인 상황!!->200
        num_imgs=int(n_train/num_shards) #250
        
        idx_shard = [i for i in range(num_shards)] #length=200
        net_dataidx_map = {i: np.array([]) for i in range(n_nets)} #length=100
#         net_dataidx_map = {}        
        idxs = np.arange(num_shards*num_imgs) #length=n_train
        labels = np.array(y_train)#리스트에 각 data idx 별로 어떤 숫자인지 label기재!!

        # sort labels
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]#label이 0부터 9까지 monotone increasing하게 idx 재배정
        idxs = idxs_labels[0, :]#label이 0부터 9까지 monotone increasing하게 idx 재배정

        # divide and assign 2 shards/client(client 100명이라서)
        for i in range(n_nets):
            rand_set = set(np.random.choice(idx_shard, int(class_num*ratio_shard), replace=False))#0~199번쨰 idx shard 중 없앨 class_num*ratio_shard=2개의 idx 추린다.#num shard=200, num of clinet: 100이라서 정확히 모든 데이터가 쪼개어져 배정.
            idx_shard = list(set(idx_shard) - rand_set)#2개 없앤 것 반영.각 shard는 동일한 label을 갖게 된다!!
            for rand in rand_set:#i번째 client에 class_num*ratio_shard=2개의 shard를 배정해 600개의 image 준다!1=> non-iid
                net_dataidx_map[i] = np.concatenate(
                    (net_dataidx_map[i].astype(int), idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
        

    elif partition == "hetero":
        min_size = 0
        K = class_num #number of class, 10
        N = y_train.shape[0] # 50000
        logging.info("N = " + str(N))
        net_dataidx_map = {}# 각 client(key)마다  몇번째 index data를 할당할지(list)에 대한 것은 value로 지정!!

        while min_size < 10:#이게 안되면 idx_batch는 초기화ㅣ!!
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):#각 label에 대해 진행!!
                idx_k = np.where(y_train == k)[0]#[0] 없어도 될 것 같음!!
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))#dirichlet(alpha, alpha, ...alpha), alpha 작을수록 heterogenioty 증가!!, proportions는 client 수 차원의  확률 벡터!!
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])#True=1, False=0
                proportions = proportions / proportions.sum()#0, 1/number of selected client value만 가질 것임!!
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]#np.cumsum은 누적합 의미(cdf같은 개념)=>monotone increasing. [:-1}은 맨 마지막 element인 1*len(idx_k) 뺴고 다 불러오는 것!! client 개수-1개의 차원 갖고 monotone increasing
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]#np.split(idx_k, proportions)는 idx_k를 proportions기준으로 분리!!
                #각 client 기존 index에서 k본째 class data를 지정된 proportion만큼 추가!!
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])#class가 clustering되게 하지 않기 위함!!
            net_dataidx_map[j] = idx_batch[j]#각 client j마다 참고할 data idx list 저장한다!! idx_batch = [[] for _ in range(n_nets)] 틀에 저장!!

    elif partition == "hetero-fix":#안봐도 될 것 같음!!
        dataidx_map_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/net_dataidx_map.txt'
        net_dataidx_map = read_net_dataidx_map(dataidx_map_file_path)

    if partition == "hetero-fix":#안봐도 될 것 같음!!
        distribution_file_path = './data_preprocessing/non-iid-distribution/CIFAR10/distribution.txt'
        traindata_cls_counts = read_data_distribution(distribution_file_path)
    else:
        traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map)#반드시 실행되는 상황

    return X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts

"""-------------------------------------------------------------------------------------------------"""

# for centralized training
def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):#dataidxs가 None이면 전체 데이타!!
    return get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs)

def get_dataloader_CIFAR10(datadir, train_bs, test_bs, dataidxs=None):#train data에만 dataidxs 적용!!
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()
    #local에게 부여할 데이타 지정!!
    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)#dataidxs가 key!!
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)#전체 test data!!

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)#data는 torch.utils.data
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl
"""----------------------------------------------------------------------------------------------------------------"""
# for local devices
# 이 두개는 local device가 test data마저 다르게 가질 떄를 짠 것으로 이용되지 않음!!
def get_dataloader_test(dataset, datadir, train_bs, test_bs, dataidxs_train, dataidxs_test):#dataidxs가 key!!
    return get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train, dataidxs_test)

def get_dataloader_test_CIFAR10(datadir, train_bs, test_bs, dataidxs_train=None, dataidxs_test=None):
    dl_obj = CIFAR10_truncated

    transform_train, transform_test = _data_transforms_cifar10()
    #local에게 부여할 데이타 지정!!
    train_ds = dl_obj(datadir, dataidxs=dataidxs_train, train=True, transform=transform_train, download=True)#dataidxs가 key!!
    test_ds = dl_obj(datadir, dataidxs=dataidxs_test, train=False, transform=transform_test, download=True)#dataidxs가 key!!

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=True)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=True)

    return train_dl, test_dl
"""----------------------------------------------------------------------------------------------------------------"""
def load_partition_data_distributed_cifar10(process_id, dataset, data_dir, partition_method, partition_alpha,
                                            client_number, batch_size):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha)#안쓰이는 것 같음!!
    class_num = len(np.unique(y_train))
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])

    # get global test data
    if process_id == 0:
        train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)
        logging.info("train_dl_global number = " + str(len(train_data_global)))
        logging.info("test_dl_global number = " + str(len(test_data_global)))
        train_data_local = None
        test_data_local = None
        local_data_num = 0
    else:
        # get local dataset
        dataidxs = net_dataidx_map[process_id - 1]
        local_data_num = len(dataidxs)
        logging.info("rank = %d, local_sample_number = %d" % (process_id, local_data_num))
        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("process_id = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            process_id, len(train_data_local), len(test_data_local)))
        train_data_global = None
        test_data_global = None
    return train_data_num, train_data_global, test_data_global, local_data_num, train_data_local, test_data_local, class_num


def load_partition_data_cifar10(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size, ratio_shard):
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(dataset,
                                                                                             data_dir,
                                                                                             partition_method,
                                                                                             client_number,
                                                                                             partition_alpha, ratio_shard)
    #여기까지는 data augmentation 안먹임!!
    class_num = len(np.unique(y_train))#task의 라벨이 몇개인지
    logging.info("traindata_cls_counts = " + str(traindata_cls_counts))
    train_data_num = sum([len(net_dataidx_map[r]) for r in range(client_number)])#모든 client의 local data 수 합한 것

    train_data_global, test_data_global = get_dataloader(dataset, data_dir, batch_size, batch_size)#global train, test의 dataloader형태 output
    logging.info("train_dl_global number = " + str(len(train_data_global)))
    logging.info("test_dl_global number = " + str(len(test_data_global)))
    test_data_num = len(test_data_global)#global test set 데이타 갯수

    # get local dataset
    data_local_num_dict = dict()#각 client마다 몇 개의 train data를 가지고 있는지 기재
    train_data_local_dict = dict()#client의  train dataloader들을 저장한 dictionary
    test_data_local_dict = dict()#client의  train dataloader들을 저장한 dictionary, 모든 client의 test data loader는 같게 배정

    for client_idx in range(client_number):#각 client_idx마다 실행!!
        dataidxs = net_dataidx_map[client_idx]
        local_data_num = len(dataidxs)
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d" % (client_idx, local_data_num))

        # training batch size = 64; algorithms batch size = 32
        train_data_local, test_data_local = get_dataloader(dataset, data_dir, batch_size, batch_size,
                                                 dataidxs)
        logging.info("client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d" % (
            client_idx, len(train_data_local), len(test_data_local)))
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_local
    return train_data_num, test_data_num, train_data_global, test_data_global, \
           data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num
