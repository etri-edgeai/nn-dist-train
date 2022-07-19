import os
import logging
import copy
import random

import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from fedml_api.utils.utils_condense import TensorDataset, TensorDataset_client_idx

# Model trainer for ensemble distillation

class MyModelTrainer(ModelTrainer):
    
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
        
    def save_model(self, save_dir, round_idx, model_type='client'):
        model = self.model
        model_file_name = model_type + str(round_idx) + '_%04d'%(int(self.id))
        save_path = os.path.join(save_dir, model_file_name)
        torch.save(model.cpu().state_dict(), save_path)

    def load_client_models(self, device, args):
        selected_client_indexes = self.client_indexes
        flist = os.listdir(self.save_model_dir)
        save_paths = []
        for fname in flist:
            selected_client = fname.split('_')[-1]
            if int(selected_client) in selected_client_indexes.tolist():
                save_paths.append(os.path.join(self.save_model_dir ,fname))
        assert len(save_paths) == len(selected_client_indexes)
        
        model = copy.deepcopy(self.model)
        client_models = []
        
        with torch.no_grad():
            for path in save_paths:
                model.cpu().load_state_dict(torch.load(path))
                model.eval()
                model = model.to(device)
                client_models.append(copy.deepcopy(model))
        self.client_models=client_models
        
    
    def get_logits_from_clients(self, image, device ,args):
        # Load model params in client instance
        # Make model class
        # Feedforward model in eval mode
        # Need Current round selected clients
        
        data_num = image.size(0)
        avg_logits = torch.zeros(data_num, self.class_num, device=device)
        
        with torch.no_grad():
            for client_model in self.client_models:
                client_model.eval()
                image = image.to(device)
                client_model = client_model.to(device)
                avg_logits += client_model(image)

        avg_logits /= len(self.client_models)
        avg_logits = F.softmax(avg_logits, dim=1)
        return avg_logits.detach()
    
    def get_logits_from_one_client(self, image, client_model, device ,args):
        # Load model params in client instance
        # Make model class
        # Feedforward model in eval mode
        # Need Current round selected clients
        
        data_num = image.size(0)
        one_logits = torch.zeros(data_num, self.class_num, device=device)
        
        with torch.no_grad():
                client_model.eval()
                image = image.to(device)
                client_model = client_model.to(device)
                one_logits += client_model(image)

        one_logits = F.softmax(one_logits, dim=1)
        return one_logits.detach()
    

    def get_logits_from_one_client2(self, image, client_idxs, device ,args):
        
        # get logit from client data each
        data_num = image.size(0)
        one_logits = torch.zeros(data_num, self.class_num, device=device)
        
        with torch.no_grad():
                for idx, (img, client_idx) in enumerate(zip(image, client_idxs)):
                    ######## get client model according to client_idx
                    c_idx = np.where(self.client_indexes == client_idx.item())[0]
                    client_model = self.client_models[c_idx[0]]
                    client_model.eval()
                    img = img.unsqueeze(0)
                    img = img.to(device)
                    client_model = client_model.to(device)
                    one_logits[idx] = client_model(img)
        one_logits = F.softmax(one_logits, dim=1)
        return one_logits.detach()
    
    # Online Training
    def train(self, train_data, val_data, device, args):
        
        # init client_model
        self.load_client_models(device, args)

        model = self.model
        model.to(device)
        model.train()

        # train and update
        criterion = nn.KLDivLoss(reduction='batchmean').to(device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.server_lr)
        scheduler = CosineAnnealingLR(optimizer, args.server_steps)
  
        epoch = 0
        epoch_loss = []
        curr_step = 0
        patience_step = 0
        curr_val_acc = 0
        best_val_acc = 0
        
        while curr_step < args.server_steps and patience_step < args.server_patience_steps:
            batch_loss = []
            with tqdm(train_data, unit="Step") as tstep:
                for batch_idx, (x, labels) in enumerate(tstep):
                    tstep.set_description(f"Step {curr_step}")

                    if curr_step < args.server_steps and patience_step < args.server_patience_steps:
                        model.train()
                        x, _ = x.to(device), labels.to(device)
                        optimizer.zero_grad()
                        model.zero_grad()
                        log_prob = model(x)
                        log_prob = F.log_softmax(log_prob, dim=1)
                        
                        # Get average logits from clients
                        avg_logits = self.get_logits_from_clients(x, device, args)

                        loss = criterion(log_prob, avg_logits.detach())
                        loss.backward()

                        optimizer.step()
                        batch_loss.append(loss.item())

                        curr_step += 1
                        patience_step += 1

                        ## Evaluate
                        if val_data:
                            if curr_step % 100 == 0 :
                                curr_val_acc = self.validate(val_data, device, args)
                                if curr_val_acc > best_val_acc:
                                    best_val_acc = curr_val_acc
                                    patience_step = 0
                        scheduler.step()

                        tstep.set_postfix(val_acc=curr_val_acc, best_val_acc=best_val_acc, step_loss=loss.item())

                    else:
                        # If val_acc plateaus or reaches server_steps
                        break
                        
        del self.client_models
        return best_val_acc
    
    def train_condense_only_one2(self, condensed_data_dict, val_data, device, args):
    
        # one by one load client
        # train model
        # Client idx를 주면 avg logits이 아니라 그냥 logit를 돌려주도록 하는 함수 get_logit_from one client
        # 아래 condense_soft 메서드 반복문 돌기

        # init client_models
        self.load_client_models(device, args)
        model = self.model
        model.to(device)
        model.train()
        
        # train and update
        criterion = nn.KLDivLoss(reduction='batchmean').to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.condense_lr)
        scheduler = CosineAnnealingLR(optimizer, args.condense_server_steps)
        
        syn_data = []
        syn_label = []
        syn_client_idx = []
        for c_idx, c in enumerate(self.client_indexes):
            num_data = condensed_data_dict[c][0].size(0)
            syn_data.append(condensed_data_dict[c][0])
            syn_label.append(condensed_data_dict[c][1])
            syn_client_idx.extend([c] * num_data)
        
        syn_data = torch.cat(syn_data)
        syn_label = torch.cat(syn_label)
        condensed_ds = TensorDataset_client_idx(syn_data, syn_label, syn_client_idx)
        condensed_dl = torch.utils.data.DataLoader(condensed_ds, batch_size=args.condense_batch_size, shuffle=True)    

        epoch_loss = []
        curr_step = 0
        patience_step = 0
        curr_val_acc = 0
        best_val_acc = 0
        while curr_step < args.condense_server_steps and patience_step < args.condense_patience_steps: ## Temp steps
        #while curr_step < 500 and patience_step < 200:  ## Temp steps
            batch_loss = []
            with tqdm(condensed_dl) as tstep:
                for batch_idx, (x, labels, client_idxs) in enumerate(tstep):
                    tstep.set_description(f"Step {curr_step}")
                    if curr_step < args.condense_server_steps and \
                    patience_step < args.condense_patience_steps:
                        model.train()
                        x, _ = x.to(device), labels.to(device)
                        optimizer.zero_grad()
                        model.zero_grad()
                        log_prob = model(x)
                        log_prob = F.log_softmax(log_prob, dim=1)

                        # Get average logits from one client
                        #avg_logits = self.get_logits_from_one_client(x, client_model, device, args)
                        avg_logits = self.get_logits_from_one_client2(x, client_idxs, device, args)

                        loss = criterion(log_prob, avg_logits.detach())
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        batch_loss.append(loss.detach().clone().item())
                        curr_step += 1
                        patience_step += 1

                        ## Evaluate
                        if val_data:
                            if curr_step % 50 == 0 :  ## Temp
                                curr_val_acc = self.validate(val_data, device, args)
                                if curr_val_acc > best_val_acc:
                                    best_val_acc = curr_val_acc
                                    patience_step = 0

                        tstep.set_postfix(val_acc=curr_val_acc,
                                          best_val_acc=best_val_acc,
                                          step_loss=loss.item())

                    else :
                        break

            #logging.info("Epoch loss : {}".format(sum(batch_loss) / len(batch_loss)))
        del self.client_models
        return best_val_acc
    def train_condense_only_one(self, condensed_data_dict, val_data, device, args):
    
        # one by one load client
        # train model
        # Client idx를 주면 avg logits이 아니라 그냥 logit를 돌려주도록 하는 함수 get_logit_from one client
        # 아래 condense_soft 메서드 반복문 돌기

        # init client_models
        self.load_client_models(device, args)
        model = self.model
        model.to(device)
        model.train()
        
        # train and update
        criterion = nn.KLDivLoss(reduction='batchmean').to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.condense_lr)
        scheduler = CosineAnnealingLR(optimizer, args.condense_server_steps)
        

        for c_idx, c in enumerate(self.client_indexes):
            client_model = self.client_models[c_idx]
            condensed_ds = TensorDataset(condensed_data_dict[c][0], condensed_data_dict[c][1])
            condensed_dl = torch.utils.data.DataLoader(condensed_ds, batch_size=6, shuffle=True) ## Temp batch size

            epoch_loss = []
            curr_step = 0
            patience_step = 0
            curr_val_acc = 0
            best_val_acc = 0
            while curr_step < args.condense_server_steps and patience_step < args.condense_patience_steps: ## Temp steps
            #while curr_step < 400 and patience_step < 200:  ## Temp steps
                batch_loss = []
                with tqdm(condensed_dl) as tstep:
                    for batch_idx, (x, labels) in enumerate(tstep):
                        tstep.set_description(f"Step {curr_step}")
                        if curr_step <  args.condense_server_steps and \
                        patience_step < args.condense_patience_steps :
                            model.train()
                            x, _ = x.to(device), labels.to(device)
                            optimizer.zero_grad()
                            model.zero_grad()
                            log_prob = model(x)
                            log_prob = F.log_softmax(log_prob, dim=1)

                            # Get average logits from one client
                            avg_logits = self.get_logits_from_one_client(x, client_model, device, args)

                            loss = criterion(log_prob, avg_logits.detach())
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                            batch_loss.append(loss.detach().clone().item())
                            curr_step += 1
                            patience_step += 1

                            ## Evaluate
                            if val_data:
                                if curr_step % 50 == 0 :  ## Temp
                                    curr_val_acc = self.validate(val_data, device, args)
                                    if curr_val_acc > best_val_acc:
                                        best_val_acc = curr_val_acc
                                        patience_step = 0

                            tstep.set_postfix(val_acc=curr_val_acc,
                                              best_val_acc=best_val_acc,
                                              step_loss=loss.item())

                        else :
                            break

                #logging.info("Epoch loss : {}".format(sum(batch_loss) / len(batch_loss)))
        del self.client_models
        return best_val_acc
        
    ####################
    def train_wth_public_condense_soft(self, condensed_data, unlabeled_dataloader, val_data, device, args):
        # make dataset and dataloader
        # train with cross entropy
        condensed_ds = TensorDataset(condensed_data[0], condensed_data[1])
        condensed_dl = torch.utils.data.DataLoader(condensed_ds, batch_size=args.condense_batch_size, shuffle=True)
    
        # init client_model
        self.load_client_models(device, args)
        
        model = self.model
        model.to(device)
        model.train()
        
        # train and update
        criterion = nn.KLDivLoss(reduction='batchmean').to(device)
        
        if args.condense_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.condense_lr)
        elif args.condense_optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.condense_lr)
        else :
            raise ValueError("{} not defined".format(args.condense_optimizer))
        scheduler = CosineAnnealingLR(optimizer, args.condense_server_steps)

        epoch_loss = []
        curr_step = 0
        patience_step = 0
        curr_val_acc = 0
        best_val_acc = 0
        while curr_step < args.condense_server_steps and \
        patience_step < args.condense_patience_steps:
            unlabeled_iterator = iter(unlabeled_dataloader)
            batch_loss = []
            with tqdm(condensed_dl) as tstep:
                for batch_idx, (x, labels) in enumerate(tstep):
                    tstep.set_description(f"Step {curr_step}")
                    if curr_step < args.condense_server_steps and \
                    patience_step < args.condense_patience_steps:
                        
                        try : 
                            x2, labels2 = next(unlabeled_iterator)
                        except StopIteration:
                            unlabeled_iterator = iter(unlabeled_dataloader)
                            x2, labels2 = next(unlabeled_iterator)
                            
                        x, _ = x.to(device), labels.to(device)
                        x2, _ = x2.to(device), labels.to(device)
                        ## Combine x, x2
                        combined_x = torch.cat([x, x2])
                            
                        
                        model.train()                        
                        optimizer.zero_grad()
                        model.zero_grad()
                        log_prob = model(combined_x)
                        log_prob = F.log_softmax(log_prob, dim=1)

                        # Get average logits from clients
                        avg_logits = self.get_logits_from_clients(combined_x, device, args)
                        
                        loss = criterion(log_prob, avg_logits.detach())
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        batch_loss.append(loss.detach().clone().item())
                        curr_step += 1
                        patience_step += 1

                        ## Evaluate
                        if val_data:
                            if curr_step % 100 == 0 :
                                curr_val_acc = self.validate(val_data, device, args)
                                if curr_val_acc > best_val_acc:
                                    best_val_acc = curr_val_acc
                                    patience_step = 0

                        tstep.set_postfix(val_acc=curr_val_acc,
                                          best_val_acc=best_val_acc, 
                                          step_loss=loss.item())

                    else :
                        break
                
                #logging.info("Epoch loss : {}".format(sum(batch_loss) / len(batch_loss)))
        del self.client_models
        return best_val_acc
    
    def train_wth_condense_soft(self, condensed_data, val_data, device, args):
        # make dataset and dataloader
        # train with cross entropy
        condensed_ds = TensorDataset(condensed_data[0], condensed_data[1])
        condensed_dl = torch.utils.data.DataLoader(condensed_ds, batch_size=args.condense_batch_size, shuffle=True)
    
        # init client_model
        self.load_client_models(device, args)
        
        model = self.model
        model.to(device)
        model.train()
        
        # train and update
        criterion = nn.KLDivLoss(reduction='batchmean').to(device)
        
        if args.condense_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.condense_lr)
        elif args.condense_optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.condense_lr)
        else :
            raise ValueError("{} not defined".format(args.condense_optimizer))
        scheduler = CosineAnnealingLR(optimizer, args.condense_server_steps)

        epoch_loss = []
        curr_step = 0
        patience_step = 0
        curr_val_acc = 0
        best_val_acc = 0
        while curr_step < args.condense_server_steps and \
        patience_step < args.condense_patience_steps:
            batch_loss = []
            with tqdm(condensed_dl) as tstep:
                for batch_idx, (x, labels) in enumerate(tstep):
                    tstep.set_description(f"Step {curr_step}")
                    if curr_step < args.condense_server_steps and \
                    patience_step < args.condense_patience_steps:
                        model.train()
                        x, _ = x.to(device), labels.to(device)
                        optimizer.zero_grad()
                        model.zero_grad()
                        log_prob = model(x)
                        log_prob = F.log_softmax(log_prob, dim=1)

                        # Get average logits from clients
                        avg_logits = self.get_logits_from_clients(x, device, args)
                        
                        loss = criterion(log_prob, avg_logits.detach())
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                        batch_loss.append(loss.detach().clone().item())
                        curr_step += 1
                        patience_step += 1

                        ## Evaluate
                        if val_data:
                            if curr_step % 100 == 0 :
                                curr_val_acc = self.validate(val_data, device, args)
                                if curr_val_acc > best_val_acc:
                                    best_val_acc = curr_val_acc
                                    patience_step = 0

                        tstep.set_postfix(val_acc=curr_val_acc,
                                          best_val_acc=best_val_acc, 
                                          step_loss=loss.item())

                    else :
                        break
                
                #logging.info("Epoch loss : {}".format(sum(batch_loss) / len(batch_loss)))
        del self.client_models
        return best_val_acc
    
    def train_wth_condense(self, condensed_data, val_data, device, args):
        # make dataset and dataloader
        # train with cross entropy
        condensed_ds = TensorDataset(condensed_data[0], condensed_data[1])
        condensed_dl = torch.utils.data.DataLoader(condensed_ds, batch_size=16, shuffle=True)
        
        # init clients
        self.load_client_models(device, args)
        
        model = self.model
        model.to(device)
        model.train()
        
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        
        if args.condense_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.condense_lr)
        elif args.condense_optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.condense_lr)
        else :
            raise ValueError("{} not defined".format(args.condense_optimizer))
        scheduler = CosineAnnealingLR(optimizer, args.condense_server_steps)

          
        epoch_loss = []
        curr_step = 0
        patience_step = 0
        curr_val_acc = 0
        best_val_acc = 0
        
        while curr_step < args.condense_server_steps and \
        patience_step < args.condense_patience_steps:
            with tqdm(condensed_dl) as tstep:
                batch_loss = []
                for batch_idx, (x, labels) in enumerate(tstep):
                    tstep.set_description(f"Step {curr_step}")
                    
                    if curr_step < args.condense_server_steps and \
                    patience_step < args.server_patience_steps:
                        model.train()
                        x, labels = x.to(device), labels.to(device)
                        optimizer.zero_grad()
                        model.zero_grad()
                        log_probs = model(x)
                        loss = criterion(log_probs, labels)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        batch_loss.append(loss.detach().clone().item())
                        curr_step += 1
                        patience_step += 1

                        ## Evaluate
                        if val_data:
                            if curr_step  % 100 == 0 :
                                curr_val_acc = self.validate(val_data, device, args)
                                if curr_val_acc > best_val_acc:
                                    best_val_acc = curr_val_acc
                                    patience_step = 0
                        
                        tstep.set_postfix(val_acc=curr_val_acc, bset_val_acc=best_val_acc, step_loss=loss.item())
                    
                    else :
                        break
                    
                logging.info("Epoch loss : {}".format(sum(batch_loss) / len(batch_loss)))

        del self.client_models
        return best_val_acc
    
    
    def validate(self, val_data, device, args):
        model = self.model
        model.eval()
        correct = 0
        n_data = 0

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(val_data):
                
                x, labels = x.to(device), labels.to(device)
                output = model(x)
                predicted = torch.argmax(output, dim=1)

                batch_correct = predicted.eq(labels).sum()
                batch_num = x.size(0)

                correct += batch_correct
                n_data += batch_num

            val_acc = correct / n_data

        return val_acc.detach().item()


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

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False


class MyModelTrainer_fedmix(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
        
    def load_client_models(self, device, args):
        selected_client_indexes = self.client_indexes
        flist = os.listdir(self.save_model_dir)
        save_paths = []
        for fname in flist:
            selected_client = fname.split('_')[-1]
            if int(selected_client) in selected_client_indexes.tolist():
                save_paths.append(os.path.join(self.save_model_dir ,fname))
        assert len(save_paths) == len(selected_client_indexes)
        
        model = copy.deepcopy(self.model)
        client_models = []
        
        with torch.no_grad():
            for path in save_paths:
                model.cpu().load_state_dict(torch.load(path))
                model.eval()
                model = model.to(device)
                client_models.append(copy.deepcopy(model))
        self.client_models=client_models

    def get_logits_from_clients(self, image, device ,args):
        # Load model params in client instance
        # Make model class
        # Feedforward model in eval mode
        # Need Current round selected clients

        data_num = image.size(0)
        avg_logits = torch.zeros(data_num, self.class_num, device=device)
        
        with torch.no_grad():
            for client_model in self.client_models:
                client_model.eval()
                image = image.to(device)
                client_model = client_model.to(device)
                avg_logits += client_model(image)

        avg_logits /= len(self.client_models)
        avg_logits = F.softmax(avg_logits, dim=1)
        return avg_logits.detach()

    # Online Training
    def train(self, train_data, average_data, val_data, device, args):

        image_means, label_means = average_data[0], average_data[1]
        image_means, label_means = image_means.to(device), label_means.to(device)
        
        # init client_model
        self.load_client_models(device, args)
        
        model = self.model
        model.to(device)
        model.train()
        
        # train and update
        criterion = nn.KLDivLoss(reduction='batchmean').to(device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.server_lr)
        scheduler = CosineAnnealingLR(optimizer, args.server_steps)

        epoch = 0
        epoch_loss = []
        curr_step = 0
        patience_step = 0
        curr_val_acc = 0
        best_val_acc = 0
        
        # For mixup
        lam = args.lam
        avg_ds = TensorDataset(image_means, label_means)
        avg_loader = torch.utils.data.DataLoader(avg_ds,
                                                 batch_size=16,
                                                shuffle=True)

        while curr_step < args.server_steps and patience_step < args.server_patience_steps:
            batch_loss = []            
            with tqdm(avg_loader, unit="Step") as tstep:
                for batch_idx, (images_1, labels_1) in enumerate(tstep):
                    tstep.set_description(f"Step {curr_step}")
                    
                    if curr_step < args.server_steps and patience_step < args.server_patience_steps:                     
                        images_1 = images_1.to(device)
                        optimizer.zero_grad()
                        model.zero_grad()
                        output = model(images_1)
                        log_prob = F.log_softmax(output, dim=1)
                        # Get average logits from clients
                        avg_logits = self.get_logits_from_clients(images_1, 
                                                                  device, args)
          
                        loss = criterion(log_prob, avg_logits)
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        batch_loss.append(loss.clone().detach().item())

                        curr_step += 1
                        patience_step += 1
                        ## Evaluate
                        if val_data:
                            if curr_step % 100 == 0 :
                                curr_val_acc = self.validate(val_data, device, args)
                                if curr_val_acc > best_val_acc:
                                    best_val_acc = curr_val_acc
                                    patience_step = 0

                        tstep.set_postfix(val_acc=curr_val_acc, best_val_acc=best_val_acc, step_loss=loss.item())


                    else:
                        # If val_acc plateaus or reaches server_steps
                        break
                
                # epoch_loss = sum(batch_loss) / len(batch_loss)
                # print("Epoch Loss : " , epoch_loss)

        del self.client_models
        return best_val_acc

    def validate(self, val_data, device, args):
        model = self.model
        model.eval()
        correct = 0
        n_data = 0

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(val_data):
                x, labels = x.to(device), labels.to(device)
                output = model(x)
                predicted = torch.argmax(output, dim=1)

                batch_correct = predicted.eq(labels).sum()
                batch_num = x.size(0)

                correct += batch_correct
                n_data += batch_num

            val_acc = correct / n_data

        return val_acc.detach().item()


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

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    
class MyModelTrainer_fedmix_wth_unlabel(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_logits_from_clients(self, image, device ,args):
        # Load model params in client instance
        # Make model class
        # Feedforward model in eval mode
        # Need Current round selected clients

        selected_client_indexes = self.client_indexes
        flist = os.listdir(self.save_model_dir)
        save_paths = []
        for fname in flist:
            selected_client = fname[-1]
            if int(selected_client) in selected_client_indexes.tolist():
                save_paths.append(os.path.join(self.save_model_dir ,fname))

        # save_paths = [selected_client for selected_client in save_paths
        #               if int(selected_client[-1]) in selected_client_indexes.tolist()]
        # # Choose clients which are selected in this round
        data_num = image.size(0)
        model = copy.deepcopy(self.model)
        avg_logits = torch.zeros(data_num, self.class_num, device=device)

        with torch.no_grad():
            for path in save_paths:
                model.cpu().load_state_dict(torch.load(path))
                model.eval()

                image = image.to(device)
                model = model.to(device)
                avg_logits += model(image)

        avg_logits /= len(save_paths)
        avg_logits = F.softmax(avg_logits, dim=1)

        return avg_logits

    # Online Training
    def train(self, train_data, average_data, val_data, device, args):

        image_means, label_means = average_data[0], average_data[1]
        image_means, label_means = image_means.to(device), label_means.to(device)
        
        model = self.model
        model.to(device)
        model.train()
        
        # train and update
        criterion = nn.KLDivLoss(reduction='batchmean').to(device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.server_lr)
        scheduler = CosineAnnealingLR(optimizer, args.server_steps)

        epoch = 0
        epoch_loss = []
        curr_step = 0
        patience_step = 0
        curr_val_acc = 0
        best_val_acc = 0
        
        lam = args.lam
        while curr_step < args.server_steps and patience_step < args.server_patience_steps:
            batch_loss = []
            with tqdm(train_data, unit="Step") as tstep:
                for batch_idx, (images_1, labels_1) in enumerate(tstep):
                    tstep.set_description(f"Step {curr_step}")

                    if curr_step < args.server_steps and patience_step < args.server_patience_steps:                        
                        # with only averaged data
  
                        batch_size = labels_1.size()[0]
                        images_1, labels_1 = images_1.to(device), labels_1.to(device)
                        num_2 = label_means.size()[0]
                        idx2 = np.random.choice(range(num_2), 1, replace=False)
                        images_2, labels_2 = image_means[idx2], label_means[idx2]   
                        model.zero_grad()
                        
                        mixed_x = lam * images_1 + (1 - lam) * images_2
                        
                        output = model(mixed_x)
                        log_probs = F.log_softmax(output, dim=1)
                        
                        # Get average logits from clients, work as labels
                        avg_logits1 = self.get_logits_from_clients(images_1, 
                                                                  device, args)
                        avg_logits2 = self.get_logits_from_clients(images_2, 
                                                                  device, args)

                        loss1 = criterion(log_probs, avg_logits1)
                        loss2 = criterion(log_probs, avg_logits2)
                        
                        loss = lam * loss1 + (1-lam) * loss2
                        loss.backward()
                        # to avoid nan loss
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        optimizer.step()
                        batch_loss.append(loss.item())

                        curr_step += 1
                        patience_step += 1
                        ## Evaluate
                        if val_data:
                            if curr_step + 1 % 100 == 0 :
                                curr_val_acc = self.validate(val_data, device, args)
                                if curr_val_acc > best_val_acc:
                                    best_val_acc = curr_val_acc
                                    patience_step = 0
                        scheduler.step()

                        tstep.set_postfix(val_acc=curr_val_acc, best_val_acc=best_val_acc, step_loss=loss.item())


                    else:
                        # If val_acc plateaus or reaches server_steps
                        break

                # epoch += 1
                # epoch_loss.append(sum(batch_loss) / len(batch_loss))
                # logging.info("Server Epoch {} Validate acc {:.3f}".format(epoch, curr_val_acc.item()))

        return best_val_acc

    def validate(self, val_data, device, args):
        model = self.model
        model.eval()
        correct = 0
        n_data = 0

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(val_data):
                x, labels = x.to(device), labels.to(device)
                output = model(x)
                predicted = torch.argmax(output, dim=1)

                batch_correct = predicted.eq(labels).sum()
                batch_num = x.size(0)

                correct += batch_correct
                n_data += batch_num

            val_acc = correct / n_data

        return val_acc.detach().item()


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

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
    
class MyModelTrainer_full_logits(ModelTrainer):

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_logits_from_clients(self, image, device, args):
        # Load model params in client instance
        # Make model class
        # Feedforward model in eval mode
        # Need Current round selected clients
        # Get logits but ** don't average them ** !!
        # Don't collect all the logits from all clients, but random sample one client at a time

        selected_client_indexes = self.client_indexes
        flist = os.listdir(self.save_model_dir)
        save_paths = []
        for fname in flist:
            selected_client = fname.split('_')[-1]
            if int(selected_client) in selected_client_indexes.tolist():
                save_paths.append(os.path.join(self.save_model_dir, fname))


        # save_paths = [selected_client for selected_client in save_paths
        #               if int(selected_client[-1]) in selected_client_indexes.tolist()]
        # # Choose clients which are selected in this round
        data_num = image.size(0)
        model = copy.deepcopy(self.model)
        full_logits = np.zeros((0, self.class_num))
        num_clients = len(save_paths)
        with torch.no_grad():
            sampled_client_idx = random.randint(0, num_clients-1)
            path = save_paths[sampled_client_idx]
            model.cpu().load_state_dict(torch.load(path))
            model.eval()

            image = image.to(device)
            model = model.to(device)

            logits = model(image)
            logits = logits.detach().cpu().clone().numpy()
            full_logits = np.vstack((full_logits, logits))

        full_logits = torch.from_numpy((full_logits))
        full_logits = full_logits.to(device)
        full_logits.double()
        return full_logits

    # Online Training
    def train(self, train_data, val_data, device, args):

        model = self.model
        model.to(device)
        
        # train and update
        criterion = nn.KLDivLoss(reduction='batchmean').to(device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.server_lr)
        scheduler = CosineAnnealingLR(optimizer, args.server_steps)

        epoch = 0
        epoch_loss = []
        curr_step = 0
        patience_step = 0
        curr_val_acc = 0
        best_val_acc = 0

        while curr_step < args.server_steps and patience_step < args.server_patience_steps:
            batch_loss = []
            with tqdm(train_data, unit="Step") as tstep:
                for batch_idx, (x, labels) in enumerate(tstep):
                    tstep.set_description(f"Step {curr_step}")

                    if curr_step < args.server_steps and patience_step < args.server_patience_steps:
                        model.train()
                        x, labels = x.to(device), labels.to(device)
                        batch_size, channel, w, h = x.size()
                        # Get average logits from clients
                        full_logits = self.get_logits_from_clients(x, device, args)

                        # For full logit training from all clients
                        # x = x.unsqueeze(1).expand(-1, args.client_num_per_round, -1, -1, -1)
                        # x = x.reshape( -1 ,3 , w, h )

                        model.zero_grad()
                        output = model(x)
                        log_prob = F.log_softmax(output, dim=1, dtype=torch.float64)
                        label_prob = F.softmax(labels, dim=1)

                        loss = criterion(log_prob, full_logits)
                        loss.backward()

                        # to avoid nan loss
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        optimizer.step()
                        batch_loss.append(loss.item())

                        curr_step += 1
                        patience_step += 1

                        ## Evaluate
                        if val_data:
                            if curr_step + 1 % 100 == 0 :
                                curr_val_acc = self.validate(val_data, device, args)
                                if curr_val_acc > best_val_acc:
                                    best_val_acc = curr_val_acc
                                    patience_step = 0
                        scheduler.step()

                        tstep.set_postfix(val_acc=curr_val_acc, best_val_acc=best_val_acc, step_loss=loss.item())


                    else:
                        # If val_acc plateaus or reaches server_steps
                        break

                # epoch += 1
                # epoch_loss.append(sum(batch_loss) / len(batch_loss))
                # logging.info("Server Epoch {} Validate acc {:.3f}".format(epoch, curr_val_acc.item()))

        return best_val_acc

    # Offline training
    # def train(self, train_data, val_data, device, args):
    #
    #     model = self.model
    #     model.to(device)
    #
    #     # train and update
    #     criterion = nn.KLDivLoss(reduction='batchmean').to(device)
    #
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.server_lr,
    #                                      weight_decay=args.wd, amsgrad=True)
    #     scheduler = CosineAnnealingLR(optimizer, args.server_steps)
    #
    #     epoch = 0
    #     epoch_loss = []
    #     curr_step = 0
    #     patience_step = 0
    #     curr_val_acc = 0
    #     best_val_acc = 0
    #
    #     while curr_step < args.server_steps and patience_step < args.server_patience_steps :
    #         batch_loss = []
    #         with tqdm(train_data, unit="Step") as tstep :
    #             for batch_idx, (x, labels) in enumerate(tstep):
    #                 tstep.set_description(f"Step {curr_step}")
    #
    #                 if curr_step < args.server_steps and patience_step < args.server_patience_steps :
    #                     model.train()
    #                     x, labels = x.to(device), labels.to(device)
    #                     model.zero_grad()
    #                     output = model(x)
    #                     log_prob = F.log_softmax(output, dim=1)
    #                     label_prob = F.softmax(labels, dim=1)
    #                     loss = criterion(log_prob, labels)
    #                     loss.backward()
    #
    #                     # to avoid nan loss
    #                     torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    #
    #                     optimizer.step()
    #                     batch_loss.append(loss.item())
    #
    #                     curr_step += 1
    #                     patience_step += 1
    #
    #                     ## Evaluate
    #                     if val_data :
    #                         curr_val_acc = self.validate(val_data, device, args)
    #                         if curr_val_acc > best_val_acc:
    #                             best_val_acc = curr_val_acc
    #                             patience_step = 0
    #                     scheduler.step()
    #
    #                     tstep.set_postfix(val_acc=curr_val_acc, best_val_acc=best_val_acc, step_loss=loss.item())
    #
    #
    #                 else :
    #                     # If val_acc plateaus or reaches server_steps
    #                     break
    #
    #             # epoch += 1
    #             # epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #             # logging.info("Server Epoch {} Validate acc {:.3f}".format(epoch, curr_val_acc.item()))
    #
    #     return best_val_acc

    def validate(self, val_data, device, args):
        model = self.model
        model.eval()
        correct = 0
        n_data = 0

        with torch.no_grad():
            for batch_idx, (x, labels) in enumerate(val_data):
                x, labels = x.to(device), labels.to(device)
                output = model(x)
                predicted = torch.argmax(output, dim=1)

                batch_correct = predicted.eq(labels).sum()
                batch_num = x.size(0)

                correct += batch_correct
                n_data += batch_num

            val_acc = correct / n_data

        return val_acc.detach().item()

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

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False