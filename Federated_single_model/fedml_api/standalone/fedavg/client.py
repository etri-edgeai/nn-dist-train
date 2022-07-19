import logging


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data#정확히 client_idx에 해당하는 data만 가져온다!!`
        self.local_test_data = local_test_data#정확히 client_idx에 해당하는 data만 가져온다!!
        self.local_sample_number = local_sample_number#정확히 client_idx에 해당하는 data 개수만 가져온다!!
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer#local iteration동안 client에서 train하는 것!!

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):#client class를 update!! 이게 핵심이다. 매 round마다 client를 다르게 가져올 수 있는 이유!!
        self.client_idx = client_idx
        self.local_training_data = local_training_data#정확히 client_idx에 해당하는 data만 가져온다!!
        self.local_test_data = local_test_data#정확히 client_idx에 해당하는 data만 가져온다!!
        self.local_sample_number = local_sample_number#정확히 client_idx에 해당하는 data 개수만 가져온다!!

    def get_sample_number(self):
        return self.local_sample_number#정확히 client_idx에 해당하는 data 개수만 가져온다!!

    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)# current global model을 올린다!!
        self.model_trainer.train(self.local_training_data, self.device, self.args)#local iteration 시행!!
        weights = self.model_trainer.get_model_params()#local iteration이 끝난 후 param을 뽑아낸다!!
        return weights

    def local_test(self, b_use_test_dataset):#local_test_data=local_training_data되게 세팅해놨음!! correct data 개수, non-averaged된 test loss, test data갯수 반환한다!!
        if b_use_test_dataset:#default로  testdata에 대해서는 b_use_test_dataset=True로 지정되어 있다!!
            test_data = self.local_test_data
        else: #default로  traindata에 대해서는 b_use_test_dataset=False로 지정되어 있다!!
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics# metrics = {'test_correct': 0,'test_loss': 0,'test_total': 0}을 뽑아낸다!!
