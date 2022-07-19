import logging
import torch

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        
        if self.args.condense:
            self.syn_data = None

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number, local_noaug_train_data=None):
        self.model_trainer.id = client_idx
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
            
    def update_local_noaug_dataset(self, local_noaug_train_data):
        self.local_noaug_train_data = local_noaug_train_data
    
        
        
    def get_sample_number(self):
        return self.local_sample_number

    def get_logits(self, test_data):
        return self.model_trainer.get_logits(test_data, self.device, self.args)
                             
    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        
        if self.args.fedmix :
            self.model_trainer.train(self.local_training_data, self.average_data, self.device, self.args)
        else :
            self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights
    
    def train_condense(self, w_global, round_idx, syn_data):
        self.model_trainer.set_model_params(w_global)
        condense_data = self.model_trainer.train_condense(self.local_training_data, self.local_noaug_train_data, self.client_idx, round_idx, syn_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights, condense_data
        
    
    
    def condense(self, weights):
        self.model_trainer.set_model_params(w_global)
        self.syn_data = self.model_trainer.condense_syndata(self.local_noaug_train_data, 
                                                           self.syn_data, self.device, self.args)
        pass
    

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics

    def update_average_dataset(self, average_data) :
        self.average_data = average_data
