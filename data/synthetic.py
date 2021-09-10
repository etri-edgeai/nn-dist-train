import logging

import json
import numpy as np
import torch
import torch.utils.data as data

__all__ = ['load_federated_synthetic']


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

train_file_path = './data/scripts/synthetic_train.json'
test_file_path = './data/scripts/synthetic_test.json'

_USERS = 'users'
_USER_DATA = "user_data"
            
            
def load_federated_synthetic(args):
    logging.info("load_federated_synthetic_1_1 START")
    
    with open(train_file_path, 'r') as train_f, open(test_file_path, 'r') as test_f:
        train_data = json.load(train_f)
        test_data = json.load(test_f)
        
        client_ids_train = train_data[_USERS]
        client_ids_test = test_data[_USERS]
        client_num = len(train_data[_USERS])
        
        args.num_clients = client_num
        
        full_x_train = torch.from_numpy(np.asarray([])).float()
        full_y_train = torch.from_numpy(np.asarray([])).long()
        full_x_test = torch.from_numpy(np.asarray([])).float()
        full_y_test = torch.from_numpy(np.asarray([])).long()
        train_data_local_dict = dict()
        test_data_local_dict = dict()
        
        for i in range(len(client_ids_train)):
            train_ds = data.TensorDataset(torch.tensor(train_data[_USER_DATA][client_ids_train[i]]['x']),
                                  torch.tensor(train_data[_USER_DATA][client_ids_train[i]]['y'], dtype=torch.int64))
            test_ds = data.TensorDataset(torch.tensor(train_data[_USER_DATA][client_ids_test[i]]['x']),
                                  torch.tensor(train_data[_USER_DATA][client_ids_test[i]]['y'], dtype=torch.int64))
            train_dl = data.DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
            test_dl = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
            train_data_local_dict[i] = train_dl
            test_data_local_dict[i] = test_dl
            
            full_x_train = torch.cat((full_x_train, torch.tensor(train_data[_USER_DATA][client_ids_train[i]]['x'])), 0)
            full_y_train = torch.cat((full_y_train, torch.tensor(train_data[_USER_DATA][client_ids_train[i]]['y'], dtype=torch.int64)), 0)
            full_x_test = torch.cat((full_x_test, torch.tensor(test_data[_USER_DATA][client_ids_test[i]]['x'])), 0)
            full_y_test = torch.cat((full_y_test, torch.tensor(test_data[_USER_DATA][client_ids_test[i]]['y'], dtype=torch.int64)), 0)
            
        train_ds = data.TensorDataset(full_x_train, full_y_train)
        test_ds = data.TensorDataset(full_x_test, full_y_test)
        train_data_global = data.DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
        test_data_global = data.DataLoader(dataset=test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        train_data_num = len(train_data_global.dataset)
        test_data_num = len(test_data_global.dataset)
        data_local_num_dict = np.array([len(train_data_local_dict[i].dataset) for i in train_data_local_dict])
        class_num = 10

    client_loader = {'train': train_data_local_dict, 'test': test_data_global}
    dataset_sizes = {'train': data_local_num_dict, 'test': test_data_num}
    
    return client_loader, dataset_sizes, args