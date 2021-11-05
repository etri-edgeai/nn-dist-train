### Comments on submitted codes
We submitted the code to reproduce our results. Since our main results are on CIFAR10 and CIFAR100, the downloading code is included in PyTorch implementation. However, the FEMNIST data used in our work exceeds the storage limits (< 50MB), we did not attach it. (Will be attached after the review process)

Example setups are included in ```./config``` directory (for ResNet-8 on CIFAR-10). One can easily implement other experiments in the paper by changing described options in the config file.

The 'fedavg' and 'fedma' for data heterogeneity stands for the data is distributed in the same manner with the corresponding papers. (label imbalance & label skewed)

### Reproduce Process
- ```pip install -r requirements.txt```
- ```wandb off```
- ```python ./main.py --config_path ./config/file_name.py```


### Train setting options
Instead of parsing the arguments to the main file, we use configuration python file in the ```./config``` directory. We describe the main options to reproduce our results as follows:

1. fed_setups:
    - In model: 
        - ```name```: Model structure.
        
    - In server_params:
        - ```n_rounds```: Number of communication rounds.
        - ```n_clients```: Number of total client devices.
        - ```sample_ratio```: Sampling ratio for each round.
        
    - In local_params:
        - ```local_ep```: Number of local epoch.
        - ```local_bs```: Local batch size.
        - ```global_loss```: Loss to global weight. (ex. proximal)
        - ```global_alpha```: Loss hyperparameter.
        
    - In data_setups:
        - ```dataset_params.dataset```: Dataset to distributed. (ex. cifar10)
        - ```distribute_params.alg```: Type of data heterogeneity. (ex. fedavg)
        - ```distribute_params.max_class_num```: Maximum number of class per client. (only for fedavg)
        - ```distribute_params.dir_alpha```: Alpha for Dirichlet distribution. (only for fedma)
    
    - In criterion:
        - ```params.mode```: Loss function to train local model. (ex. LSD)
        - ```params.beta```: Hyperparameter to loss. (only for LSD, LS-NTD)
        - ```params.temp```: Temperature to softmax. (only for LSD, LS-NTD)
        - ```params.num_classes```: Number of dataset classes.

    - In optimizer:
        - ```params.lr```: Initial learning rate.
        - ```params.momentum```: Momentum to SGD (only in local training)
