import copy
import logging
import random

import numpy as np
import torch
import wandb

from fedml_api.standalone.fedntd.client import Client


class FedntdAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset 
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None #test data set이 너무 heavy할 경우에 대체하는 용도!! See _generate_validation_set.
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []#해당 round에 참여할 client index에 해당하는 Client Class들의 저장소!!(self.args.client_num_per_round개 만큼 저장!!)
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer #global 모델에서의 test 단계를 위한 것!!
        self.class_num=class_num
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer, class_num)#initialuzation setting!!

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer, class_num):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round): #임의로 round에 참여하는 수만큼에 대해 client에 대해 Client Class 생성!! 숫자는 유지하고 round 바뀔때마다 data만 수정하는 방식!!
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, class_num)
            #self.model_trainer가 들어가도 되지 않나? 싶음, 여기서 self.model_trainer로 하면 10개의 client 모두 동일한 instance의 trainer가 들어가는 상황!!이것을 피하기 위한 의도일 수도 있음!! 허나 local train하기 앞서 w_global hash를 self.model로 항상 지정해 놓고 하기에 self.model_trainer해도 상관 없을 것 같다.
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params() #initializaton
        for round_idx in range(self.args.comm_round):
            if (round_idx+1)%5==0:
                self.args.lr*=self.args.decaylr
                
            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset <-이게 핵심!!
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)#round에 참여할 client를 sampling한다!!
            logging.info("client_indexes = " + str(client_indexes))#round에 참여할 client index 입력!!

            for idx, client in enumerate(self.client_list): #client는 class!!
                # update dataset
                client_idx = client_indexes[idx]#sampling된 client의 index로 수정해서 local data 접근!!
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])#이게 핵심!!

                # train on new dataset
                w = client.train(copy.deepcopy(w_global))#local model 추출!!
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))#client가 가지고 있는 train data 개수, local 모델 정보를 튜플 형태로 append!!

            # update global weights
            w_global = self._aggregate(w_locals)#w_local에 있는 것 바탕으로 model averaging!!(다음 라운드에 client.train 에서 이용될 거임!!)
            
            #단순 라운드에서 test를 위한 설정, test loader는 동일하기에 한 클라이언트에서만 해도 됨!!
            self.model_trainer.set_model_params(w_global)

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)#당연히 client_num_per_round되는 상황!!
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)#random하게 client select한다!! #data 갯수에 비례하하게 sampling하는 것도 가능!!
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes#round_idx 때 sample된 client들 return!!

    def _generate_validation_set(self, num_samples=10000):#test set이 너무 많아서 부담스러울 경우 sampling!!
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            sample_num, param_dict=w_locals[idx]
            training_num += sample_num
            
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():#k: model layer 갯수(각 layer 별로 model weight averaging!!)
            for i in range(0, len(w_locals)):#round에 참여하는 client 갯수!!
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]#client class 하나 가져온다!! 여기 위에 업데이트를 할 것!!

        for client_idx in range(self.args.client_num_in_total):# round에 참여하는 것만이 아닌 모든 client에 대해서 실행(global)
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None: #그럴일 없음. 모두 global test loader로 통일된 상황!!
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx]) #client_idx자리에 0이 들어갔는데 크게 안중요함!! 모두 같은 testloader를 가지고 있어서!!
            
            # train data
            train_local_metrics = client.local_test(False)#testdata=self.local_training_data, self.local_training_data에서의 metric을 output 한다.
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))#average되지 않은 값
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))#average되지 않은 값
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))#average되지 않은 값

            # test data
            test_local_metrics = client.local_test(True) #testdata=self.local_test_data, self.local_test_data에서의 metric을 output 한다.
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))#average되지 않은 값
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))#average되지 않은 값
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))#average되지 않은 값

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:#cpu로 돌아가고 있는 상태만 True
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])#전체 data 중 몇개 맞췄는지, train accracy
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])#averaged train loss

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])#전체 data 중 몇개 맞췄는지, testaccuracy
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])#averaged test loss


        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

    def _local_test_on_validation_set(self, round_idx):# 쓸 일 없음, stackoverflow에서만 쓰이는 것임!!

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:#default는 None이다!!
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)
