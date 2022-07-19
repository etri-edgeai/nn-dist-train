import logging

import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

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

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def get_affinity_metrics(self, train_data, test_data, model, device, args, round_idx, epoch):
        
        #### Affinity Metric #####
        # 1. Train acc with no updates, 2 Test acc
        
        model.eval()
        
        affinity_metrics = {
            "round_idx" : round_idx,
            "client_idx" : self.id,
            "epoch" : epoch,
            "train_acc" : 0,
            "test_acc" : 0
        }
        
        
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        batch_loss = []
        batch_correct = []
        batch_num = []

        for batch_idx, (x, labels) in enumerate(train_data):
            x, labels = x.to(device), labels.to(device)
            pred = model(x)
            loss = criterion(pred, labels)
            # to avoid nan loss
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(labels).sum()

            batch_num.append(x.size(0))
            batch_loss.append(loss.item())
            batch_correct.append(correct.item())

        epoch_num = sum(batch_num)
        train_loss = sum(batch_loss) / epoch_num
        train_acc = sum(batch_correct) / epoch_num

        affinity_metrics['train_acc'] = train_acc
           
        ##### Test #####
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)

        # Update affinity metrics
        test_acc = metrics['test_correct'] / metrics['test_total']
        affinity_metrics['test_acc'] = test_acc

        logging.info("****** Affinity metrics ******")
        logging.info(affinity_metrics)
        
    
    def train_and_test(self, train_data, test_data, device, args, round_idx=None):
        
        ##### Train #####
        model = self.model
        model.to(device)
        model.train()

        #### Affinity Metric #####
        affinity_metrics = {
            "round_idx" : round_idx,
            "client_idx" : self.id,
            "epoch" : 0,
            "test_acc" : 0
        }
        
        
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(args.epochs):
            ### Affinity metrics in first epoch and last
            if epoch == 0:
                affinity_metrics = self.get_affinity_metrics(train_data, test_data, model, device, args, round_idx, epoch)
            
            model.train()
            batch_loss = []
            
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))
        
        ### Affinity metrics in last epoch
        affinity_metrics = self.get_affinity_metrics(train_data, test_data, model, device, args, round_idx, epoch)
        
        ##### Test #####
#             if epoch in [0, args.epochs-1] :
                
#                 # Evaluate only in the first and last epoch
#                 model.eval()
                
#                 metrics = {
#                     'test_correct': 0,
#                     'test_loss': 0,
#                     'test_total': 0
#                 }

#                 with torch.no_grad():
#                     for batch_idx, (x, target) in enumerate(test_data):
#                         x = x.to(device)
#                         target = target.to(device)
#                         pred = model(x)
#                         loss = criterion(pred, target)

#                         _, predicted = torch.max(pred, -1)
#                         correct = predicted.eq(target).sum()

#                         metrics['test_correct'] += correct.item()
#                         metrics['test_loss'] += loss.item() * target.size(0)
#                         metrics['test_total'] += target.size(0)
                
#                 # Update affinity metrics
#                 test_acc = metrics['test_correct'] / metrics['test_total']
                
#                 affinity_metrics['test_acc'] = test_acc
#                 affinity_metrics['epoch'] = epoch
                
#                 logging.info("****** Affinity metrics ******")
#                 logging.info(affinity_metrics)
                
    
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
