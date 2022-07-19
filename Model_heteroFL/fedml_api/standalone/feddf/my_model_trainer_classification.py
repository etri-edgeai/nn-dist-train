import logging
import os
import copy
from _collections import defaultdict

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
import numpy as np

from fedml_api.utils.utils_condense import match_loss, TensorDataset

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer

from fedml_api.utils.utils_condense import get_loops
    

class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def save_model(self, save_dir, model_type='client'):
        model = self.model
        model_file_name = model_type + '_%04d'%(int(self.id))
        save_path = os.path.join(save_dir, model_file_name)
        torch.save(model.cpu().state_dict(), save_path)
    
    '''For fedprox'''
    def flatten(self, source):
        return torch.cat([value.flatten() for value in source.values()])
        
    def train(self, train_data, device, args):
        model = self.model
        model.to(device)
        model.train()
        
        # Fed proxs
        if args.lambda_fedprox :
            W0 = {k : v.detach().clone() for k, v in model.named_parameters()}
            W0 = self.flatten(W0)
            W0 = W0.to(device)

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            
        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            batch_correct = []
            total_num = 0

            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                if args.lambda_fedprox :
                    flatten_model = self.flatten(dict(model.named_parameters()))
                    flatten_model = flatten_model.to(device)
                    loss += args.lambda_fedprox * torch.sum((W0 - flatten_model)**2)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
                _, predicted = torch.max(log_probs, -1)
                correct = predicted.eq(labels).sum()
                
                total_num += labels.size()[0]
                batch_correct.append(correct.cpu().item())
                batch_loss.append(loss.item())


            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if epoch + 1 == args.epochs :
                logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f} \tTrain acc : {:.6f}'.format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss), sum(batch_correct) /total_num))


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
    
    def train_condense(self, train_data, train_data_no_aug, client_idx, round_idx, syn_data,
                       device, args):
                
        num_classes = args.class_num
        batch_real = args.batch_size
        ipc = args.image_per_class
        channel = args.channel
        #outer_loops, _ = get_loops(ipc) 
        outer_loops = args.outer_loops
        
        ''' update model '''
        model = self.model
        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if epoch + 1 == args.epochs :
                logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f} \tTrain acc : {:.6f}'.format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss), sum(batch_correct) /total_num))


        
        ''' organize real dataset'''
        logging.info('Start Condensing')
        indices_class = [[] for c in range(num_classes)]
        images_all, labels_all = train_data_no_aug
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = images_all.to(device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            # non iid setting where there is no image
            if len(indices_class[c]) >  n:
                idx_shuffle = np.random.permutation(indices_class[c])[:n]
                return images_all[idx_shuffle]
            else :
                return None

        mean = []
        std = []
        for ch in range(channel):
            
            temp_mean = torch.mean(images_all[:, ch, :, :])
            temp_mean = temp_mean.cpu()
            temp_std = torch.std(images_all[:, ch, :, :]) ## images_all is already normalize anyway..
            temp_std = temp_std.cpu()
            
            mean.append(temp_mean)
            std.append(temp_std)
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, mean[ch], std[ch]))
            

        ''' initialize the synthetic data '''
        im_size = (images_all.size()[-2], images_all.size()[-1])
        image_syn, label_syn = syn_data[0], syn_data[1]
        if image_syn == None :
            image_syn = torch.randn(size=(num_classes*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
            label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9] 
        else :
            image_syn = image_syn.to(device)
            label_syn = label_syn.to(device)
            image_syn.requires_grad_(True)
        
        
        '''training synthetic data'''
        model.train()
        net_parameters = list(model.parameters(c))
        
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.image_lr, momentum=0.5)
        optimizer_img.zero_grad()
        loss_avg = 0
        criterion = nn.CrossEntropyLoss().to(device)
        model = self.model
        
        
        for ol in range(outer_loops):
            
#             BN_flag = False
#             BNSizePC = 16 # For batch normalization

#             for module in model.modules():
#                 #if  "BatchNorm" in module._get_name():
#                 if  "BatchNorm" in type(module).__name__:
#                     BN_flag =True

#             if BN_flag :
#                 img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)
#                                       if get_images(c, BNSizePC ) is not None])
#                 model.train() # for updating the mu, sigma of BatchNorm
#                 output_real = model(img_real)
#                 for module in model.modules():
#                     #if 'BatchNorm' in module._get_names():
#                     if  "BatchNorm" in type(module).__name__:

#                         model.eval() # fix mu and sigma for every BatchNorm Layer

            ## Update synthetic data
            loss = torch.tensor(0.0).to(device)

            for c in range(num_classes) :
                img_real = get_images(c, batch_real) # Batch size
                if img_real is None : # No class images exist in this client
                    continue 
                
                lab_real = torch.ones((img_real.shape[0], ), device=device, dtype=torch.long) * c
                output_real = model(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))
                
                random_class = np.random.randint(10)
                if args.con_rand:
                    img_syn = image_syn[random_class*ipc: (random_class+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
                    ''' Condense only neighbor classes '''
                    if args.con_rand_neighbor :
                        neighbor_num = args.con_rand_neighbor
                        # Neighbor class list
                        neighbor_list = [(c + j) % args.class_num for j in range(neighbor_num)]
                        neighbor_class = np.random.choice(neighbor_list)
                        img_syn = image_syn[neighbor_class*ipc: (neighbor_class+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
                    
                else :
                    img_syn = image_syn[c*ipc: (c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
                lab_syn = torch.ones((ipc, ), device=device, dtype=torch.long) * c                
                output_syn = model(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                
                loss += match_loss(gw_syn, gw_real, device)
            
            
            optimizer_img.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
            loss_log =  loss.detach().cpu().item()
            
            if ol % 100 == 0 :
                logging.info('Outer loop idx : {}, loss {:.6f}'.format(ol, loss_log))            
            if ol == outer_loops -1 :
                logging.info('Outer loop idx : {}, loss {:.6f}'.format(ol, loss_log))            
                save_dir = os.path.join(args.wandb_save_dir, "./condense")
                save_name = os.path.join(save_dir, 
                                         'vis_ipc{}_ol{}_c{}_r{}.png'.format(str(ipc), str(ol), str(client_idx), str(round_idx)))

                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                save_image(image_syn_vis, save_name, nrow=ipc) # Trying normalize = True/False may get better visual effects.
                torch.save(image_syn_vis, save_name.replace("png","pt"))


            '''update network with synthetic data // TODO '''
        
        
        loss_avg /= (num_classes*outer_loops)     
        
        logging.info('Condensing Complete')
            
        image_syn, label_syn =  copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
        return (image_syn, label_syn)
    

    def condense(self, train_data_no_aug, client_idx, round_idx, syn_data,
                       device, args):
        '''initial condensation'''  
        num_classes = args.class_num
        batch_real = args.batch_size
        ipc = args.image_per_class
        channel = args.channel
        #outer_loops, _ = get_loops(ipc) 
        outer_loops = args.init_outer_loops
        
        ''' update model '''
        model = copy.deepcopy(self.model)
        model.to(device)
        model.train()

        ''' organize real dataset'''
        logging.info('Start Condensing ' + str(client_idx))
        indices_class = [[] for c in range(num_classes)]
        images_all, labels_all = train_data_no_aug
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = images_all.to(device)
        labels_all = torch.tensor(labels_all.clone().detach(), dtype=torch.long, device=device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            # non iid setting where there is no image
            if len(indices_class[c]) >  n:
                idx_shuffle = np.random.permutation(indices_class[c])[:n]
                return images_all[idx_shuffle]
            else :
                return None
        
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()
                        

        mean = []
        std = []
        
        for ch in range(channel):
            temp_mean = torch.mean(images_all[:, ch, :, :])
            temp_mean = temp_mean.cpu()
            temp_std = torch.std(images_all[:, ch, :, :]) ## images_all is already normalize anyway..
            temp_std = temp_std.cpu()
            mean.append(temp_mean)
            std.append(temp_std)
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, mean[ch], std[ch]))
            
        ''' initialize the synthetic data '''
        im_size = (images_all.size()[-2], images_all.size()[-1])
        image_syn, label_syn = syn_data[0], syn_data[1]
        if image_syn == None :
            image_syn = torch.randn(size=(num_classes*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
            label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9] 
        else :
            image_syn = image_syn.to(device)
            label_syn = label_syn.to(device)
            image_syn.requires_grad_(True)
        
        
        '''training synthetic data'''
        model.train()
        net_parameters = list(model.parameters(c))
        
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.image_lr, momentum=0.5)
        optimizer_img.zero_grad()
        scheduler = CosineAnnealingLR(optimizer_img, outer_loops)

        loss_avg = 0
        criterion = nn.CrossEntropyLoss().to(device)
        
        for ol in range(outer_loops):

            ## Update synthetic data
            loss = torch.tensor(0.0).to(device)
            loss_temp = []
            for c in range(num_classes) :
                img_real = get_images(c, batch_real) # Batch size
                if img_real is None : # No class images exist in this client
                    continue 

                lab_real = torch.ones((img_real.shape[0], ), device=device, dtype=torch.long) * c
                output_real = model(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters, allow_unused=True)
                gw_real = list((_.detach().clone() for _ in gw_real))
                
                random_class = np.random.randint(10)
                if args.con_rand:
                    img_syn = image_syn[random_class*ipc: (random_class+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
                    ''' Condense only neighbor classes '''
                    if args.con_rand_neighbor :
                        neighbor_num = args.con_rand_neighbor
                        # Neighbor class list
                        neighbor_list = [(c + j) % args.class_num for j in range(neighbor_num)]
                        neighbor_class = np.random.choice(neighbor_list)
                        img_syn = image_syn[neighbor_class*ipc: (neighbor_class+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
                    
                    
                    
                else :
                    img_syn = image_syn[c*ipc: (c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
                    
                lab_syn = torch.ones((ipc, ), device=device, dtype=torch.long) * c                
                output_syn = model(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                
                loss += match_loss(gw_syn, gw_real, device)
                loss_temp.append(loss.item())
            
            
            optimizer_img.zero_grad()
            loss.requires_grad_(True)
            loss.backward()
            optimizer_img.step()
            scheduler.step()
            loss_avg += loss.item()
            loss_log =  loss.detach().cpu().item()
            
            '''model reinit'''
            if (ol) % 20 == 0 :
                model.apply(weight_reset)
                ## Reinitializaion
            
            if ol % 100 == 0 :
                logging.info('Outer loop idx : {}, loss {:.6f}'.format(ol, loss_log))            
            
            if  ol == outer_loops -1 or ol == int(outer_loops/2) :
                logging.info('Outer loop idx : {}, loss {:.6f}'.format(ol, loss_log))            
                save_dir = os.path.join(args.wandb_save_dir, "./condense")
                save_name = os.path.join(save_dir, 
                                         'vis_ipc{}_ol{}_c{}_r{}.png'.format(str(ipc), str(ol), str(client_idx), str(round_idx)))

                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                save_image(image_syn_vis, save_name, nrow=ipc) # Trying normalize = True/False may get better visual effects.
                torch.save(image_syn_vis, save_name.replace("png","pt"))


            '''update network with synthetic data // TODO '''
        
        if outer_loops:
            loss_avg /= (num_classes*outer_loops)     
        
        logging.info('Condensing Complete')
        
#         '''save wrt round'''
#         if round_idx % 10 == 0 or round_idx == args.comm_round -1:
#             save_dir = os.path.join(args.wandb_save_dir, "./condense")
#             save_name = os.path.join(save_dir, 
#                                      'vis_ipc{}_r{}_c{}.png'.format(str(ipc), str(round_idx), str(client_idx)))

#             image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
#             for ch in range(channel):
#                 image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
#             image_syn_vis[image_syn_vis<0] = 0.0
#             image_syn_vis[image_syn_vis>1] = 1.0
#             save_image(image_syn_vis, save_name, nrow=ipc) # Trying normalize = True/False may get better visual effects.
#             torch.save(image_syn_vis, save_name.replace("png","pt"))

        "Check condense mean and std and lets savoe"
        image_syn, label_syn =  copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
            
        return (image_syn, label_syn)
                

    def train_condense_alternate(self, train_data, train_data_no_aug, client_idx, round_idx, syn_data,
                       device, args):
        
        num_classes = args.class_num
        batch_real = args.batch_size
        ipc = args.image_per_class
        channel = args.channel
        outer_loops = args.outer_loops
        
        ''' update model '''
        model = self.model
        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                optimizer.zero_grad()
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if epoch + 1 == args.epochs :
                logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f} \tTrain acc : {:.6f}'.format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss), sum(batch_correct) /total_num))

        
        ''' organize real dataset'''
        logging.info('Start Condensing')
        indices_class = [[] for c in range(num_classes)]
        images_all, labels_all = train_data_no_aug
#         images_all = [torch.unsqueeze(train_data_no_aug[i][0], dim=0) for i in range(len(train_data_no_aug))]
#         labels_all = [train_data_no_aug[i][1] for i in range(len(train_data_no_aug))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = images_all.to(device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

        for c in range(num_classes):
            print('class c = %d: %d real images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        mean = []
        std = []
        for ch in range(channel):
            
            temp_mean = torch.mean(images_all[:, ch, :, :])
            temp_mean = temp_mean.cpu()
            
            temp_std = torch.std(images_all[:, ch, :, :]) ## images_all is already normalize anyway..
            temp_std = temp_std.cpu()
            
            mean.append(temp_mean)
            std.append(temp_std)
            print('real images channel %d, mean = %.4f, std = %.4f'%(ch, mean[ch], std[ch]))
            

        ''' initialize the synthetic data '''
        im_size = (images_all.size()[-2], images_all.size()[-1])
        image_syn, label_syn = syn_data[0], syn_data[1]
        if image_syn == None :
            image_syn = torch.randn(size=(num_classes*ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
            label_syn = torch.tensor([np.ones(ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9] 
        else :
            image_syn = image_syn.to(device)
            label_syn = label_syn.to(device)
        
        
        '''training synthetic data'''
        model.train()
        net_parameters = list(model.parameters(c))
        
        optimizer_img = torch.optim.SGD([image_syn, ], lr=0.05, momentum=0.5)
        optimizer_img.zero_grad()
        loss_avg = 0
        criterion = nn.CrossEntropyLoss().to(device)
        model = self.model
        
        
        for ol in range(outer_loops):
#             BN_flag = False
#             BNSizePC = 16 # For batch normalization

#             for module in model.modules():
#                 #if  "BatchNorm" in module._get_name():
#                 if  "BatchNorm" in type(module).__name__:
#                     BN_flag =True

#             if BN_flag :
#                 img_real = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)])
#                 model.train() # for updating the mu, sigma of BatchNorm
#                 output_real = model(img_real)
#                 for module in model.modules():
#                     #if 'BatchNorm' in module._get_names():
#                     if  "BatchNorm" in type(module).__name__:

#                         model.eval() # fix mu and sigma for every BatchNorm Layer



            ## Update synthetic data
            loss = torch.tensor(0.0, requires_grad=True).to(device)

            for c in range(num_classes) :
                img_real = get_images(c, 256) # Batch size
                lab_real = torch.ones((img_real.shape[0], ), device=device, dtype=torch.long) * c
                img_syn = image_syn[c*ipc: (c+1)*ipc].reshape((ipc, channel, im_size[0], im_size[1]))
                lab_syn = torch.ones((ipc, ), device=device, dtype=torch.long) * c

                output_real = model(img_real)
                loss_real = criterion(output_real, lab_real)
                gw_real = torch.autograd.grad(loss_real, net_parameters)
                gw_real = list((_.detach().clone() for _ in gw_real))

                output_syn = model(img_syn)
                loss_syn = criterion(output_syn, lab_syn)
                gw_syn = torch.autograd.grad(loss_syn, net_parameters)
                gw_syn = list((_.detach().clone() for _ in gw_syn))

                loss += match_loss(gw_syn, gw_real, device)
            
            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()
            loss_log =  loss.detach().cpu().item()
            logging.info('Outer loop loss {:.6f}'.format(loss_log))

            '''update network with synthetic data // TODO '''
        
        
        loss_avg /= (num_classes*outer_loops)     
        
        logging.info('Start Condensing')

        logging.info('Condense complete')
        if round_idx % 20 == 0:
            save_dir = os.path.join(args.wandb_save_dir, "./condense")
            save_name = os.path.join(save_dir, 
                                     'vis_ipc{}_r{}_c{}.png'.format(str(ipc), str(round_idx), str(client_idx)))

            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            save_image(image_syn_vis, save_name, nrow=ipc) # Trying normalize = True/False may get better visual effects.
            torch.save(image_syn_vis, save_name.replace("png","pt"))

            image_syn, label_syn =  copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())
        return (image_syn, label_syn)
        
        
    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
