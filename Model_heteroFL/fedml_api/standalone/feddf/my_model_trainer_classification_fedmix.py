import logging
import os

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def save_model(self, save_dir, model_type='client'):
        model = self.model
        model_file_name = model_type + '_' + str(self.id)
        save_path = os.path.join(save_dir, model_file_name)
        torch.save(model.cpu().state_dict(), save_path)

    def train(self, train_data, average_data, device, args):
        
        images_means, labels_means = average_data[0], average_data[1]
        images_means, labels_mean = images_means.to(device), labels_means.to(device)
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

        lam = args.lam
        epoch_loss = []
        print("Start fedmix training")
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (images_1, labels_1) in enumerate(train_data):
                batch_size = labels_1.size()[0]
                images_1, labels_1 = images_1.to(device), labels_1.to(device)
                num_2 = labels_means.size()[0]
                idx2 = np.random.choice(range(num_2), 1, replace=False)
                images_2, labels_2 = images_means[idx2], labels_means[idx2]
                model.zero_grad()
                images_2_ = images_2.repeat(batch_size, 1, 1, 1)

                images_1.requires_grad_(True)
                log_probs = model((1-lam) * images_1)
                jacobian = torch.autograd.grad(outputs=log_probs[:,labels_1].sum(), inputs=images_1, retain_graph=True)[0].view(batch_size,1,-1)
                loss1 = (1-lam) * criterion(log_probs, labels_1)
                loss2 = (1-lam) * lam * torch.mean(torch.bmm(jacobian, images_2_.view(batch_size,-1,1)))
                for i in range(args.class_num):
                    if labels_2[0,i] > 0:
                        labels_2_ = i * torch.ones_like(labels_1).to(device)
                        loss1 = loss1 + labels_2[0,i] * lam * criterion(log_probs, labels_2_)
                loss = loss1 + loss2                 

                
#                 x, labels = x.to(device), labels.to(device)
#                 model.zero_grad()
#                 log_probs = model(x)
#                 loss = criterion(log_probs, labels)
                
                
                
                loss.backward()
                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
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
    
    def get_logits(self, test_data, device, args):
        model = self.model
        model.to(device)
        model.eval()
        
        full_logits = torch.from_numpy(np.asarray([])).float()
        full_logits = full_logits.to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                logits = model(x)
                full_logits = torch.cat((full_logits, logits), 0)
        return full_logits
                

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
