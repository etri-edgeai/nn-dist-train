import logging

import torch
from torch import nn

import copy

from fedml_api.standalone.fedntd.criterion import *


try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):# 각 client마다 할당됨!!
        
    def get_model_params(self):  # local 모델을 보내기 위함 !!
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):  # global model을 local에서 업로드 하기 위함!!
        self.model.load_state_dict(model_parameters)
        
        

    def train(self, train_data, device, args, class_num):
        dg_model = copy.deepcopy(self.model)
        dg_model.to(device)
        model = self.model
        model.to(device)
        model.train()

        # train and update
        #criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        

                        
        train_criterion= NTD_Loss(class_num, args.tau, args.beta)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs, t_logits = model(x), dg_model(x)                    
                        
                loss = train_criterion(log_probs,labels,t_logits)
                #                 print(args.mu/2 * self.difference_models_norm_2(model,model_0))
                #loss += args.mu / 2 * self.difference_models_norm_2(model, model_0)

                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
            
            
            
      
    

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)  # max, argmax are output!!
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:  # 안쓰이는 것 같음!!
        return False
