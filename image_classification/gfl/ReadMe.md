# Global Federated Learning (GFL) Algorithms Codebase

This repository is designed mainly for research, and pytorch implementation of GFL algorithms.  

We implemented various FL algorithms in our framework: 
- **FedAVG**, **FedProx**,  **Scaffold**, **MOON**, **FedNTD**, **FedEXP**, **FedSOL**, **FedBABU**, **SphereFed**, **FedETF**, **FedGELA**, and **FedDr+ (Ours)**.
## Installation

First check that the requirements are satisfied:</br>
Python 3.7.16</br>
pytorch 1.12.0</br>
torchvision 0.13.0</br>
numpy 1.21.5</br>
matplotlib 3.5.3</br>
Pillow 9.4.0</br>



The logs are uploaded to the wandb server. If you do not have a wandb account, just install and use as offline mode.
```
pip install wandb
wandb off
```
## Data & Model

We run GFL algorithms experiments on (CIFAR-10, VGG-11), (CIFAR-100, MobileNet), and (ImageNet-100, ResNet-18).

Before starting the implementation, first download CIFAR-10 and CIFAR-100 using the following method:

```
sh generate_cifar.sh
```

For the ImageNet-100 dataset, the code was executed by directly downloading it from the following Kaggle link:

[ImageNet-100 Dataset on Kaggle](https://www.kaggle.com/datasets/ambityga/imagenet100)

## How to Run Codes?

The configuration skeleton for each algorithm is in `./config/*.json`. 
- `python ./main.py --config_path ./config/algorithm_name.json` conducts the experiment with the default setups.

There are two ways to change the configurations:
1. Change (or Write a new one) the configuration file in `./config` directory with the above command.
2. Use parser arguments to overload the configuration file.
- `--dataset_name`: name of the datasets (e.g., `mnist`, `cifar10`, `cifar100` or `cinic10`).
  - for cinic-10 datasets, the data should be downloaded first using `./data/cinic10/download.sh`.
- `--n_clients`: the number of total clients (default: 100).
- `--batch_size`: the size of batch to be used for local training. (default: 50)
- `--partition_method`: non-IID partition strategy (e.g. `sharding`, `lda`).
- `--partition_s`: shard per user (only for `sharding`).
- `--partition_alpha`: concentration parameter alpha for latent Dirichlet Allocation (only for `lda`).
- `--model_name`: model architecture to be used (e.g., `fedavg_mnist`, `fedavg_cifar`, or `mobile`).
- `--n_rounds`: the number of total communication rounds. (default: `200`)
- `--sample_ratio`: fraction of clients to be ramdonly sampled at each round (default: `0.1`)
- `--local_epochs`: the number of local epochs (default: `5`).
- `--lr`: the initial learning rate for local training (default: `0.01`)
- `--momentum`: the momentum for SGD (default: `0.9`).
- `--wd`: weight decay for optimization (default: `1e-5`)
- `--algo_name`: algorithm name of the experiment (e.g., `fedavg`, `fedntd`)
- `--seed`: random seed


## Script for running example

We provide example implementation scripts for CIFAR-10 and CIFAR-100.

###  (CIFAR10, VGG11) Experiment
- FedAVG
	- Shard Setting
		- Run `sh fedavg_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh fedavg_cifar10_lda.sh`.

- FedProx 
	- Shard Setting
		- Run `sh fedprox_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh fedprox_cifar10_lda.sh`.
- Scaffold
	- Shard Setting
		- Run `sh scaffold_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh scaffold_cifar10_lda.sh`.		

- MOON
	- Shard Setting
		- Run `sh moon_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh moon_cifar10_lda.sh`.		

- FedNTD 
	- Shard Setting
		- Run `sh fedntd_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh fedntd_cifar10_lda.sh`.

- FedEXP
	- Shard Setting
		- Run `sh fedexp_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh fedexp_cifar10_lda.sh`.

- FedSOL
	- Shard Setting
		- Run `sh fedsol_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh fedsol_cifar10_lda.sh`.
		
- FedBABU
	- Shard Setting
		- Run `sh fedbabu_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh fedbabu_cifar10_lda.sh`.

- SphereFed
	- Shard Setting
		- Run `sh spherefed_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh spherefed_cifar10_lda.sh`.

- FedETF
	- Shard Setting
		- Run `sh fedetf_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh fedetf_cifar10_lda.sh`.	
	
- FedGELA
	- Shard Setting
		- Run `sh fedgela_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh fedgela_cifar10_lda.sh`.	
- FedDr+
	- Shard Setting
		- Run `sh feddr_cifar10_shard.sh`.
	- LDA Setting
		- Run `sh feddr_cifar10_lda.sh`.
		

###  (CIFAR100, mobileNet) Experiment
- FedAVG
	- Shard Setting
		- Run `sh fedavg_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh fedavg_cifar100_lda.sh`.

- FedProx 
	- Shard Setting
		- Run `sh fedprox_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh fedprox_cifar100_lda.sh`.
- Scaffold
	- Shard Setting
		- Run `sh scaffold_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh scaffold_cifar100_lda.sh`.		

- MOON
	- Shard Setting
		- Run `sh moon_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh moon_cifar100_lda.sh`.		

- FedNTD 
	- Shard Setting
		- Run `sh fedntd_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh fedntd_cifar100_lda.sh`.

- FedEXP
	- Shard Setting
		- Run `sh fedexp_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh fedexp_cifar100_lda.sh`.

- FedSOL
	- Shard Setting
		- Run `sh fedsol_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh fedsol_cifar100_lda.sh`.
		
- FedBABU
	- Shard Setting
		- Run `sh fedbabu_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh fedbabu_cifar100_lda.sh`.

- SphereFed
	- Shard Setting
		- Run `sh spherefed_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh spherefed_cifar100_lda.sh`.

- FedETF
	- Shard Setting
		- Run `sh fedetf_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh fedetf_cifar100_lda.sh`.

- FedGELA
	- Shard Setting
		- Run `sh fedgela_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh fedgela_cifar100_lda.sh`.	

- FedDr+
	- Shard Setting
		- Run `sh feddr_cifar100_shard.sh`.
	- LDA Setting
		- Run `sh feddr_cifar100_lda.sh`.



## FL Scenario

We utilized momentum SGD with an initial learning rate of 0.01, setting the momentum to 0.9, and applying a weight decay of 1e-5. All experiments were carried out over 320 rounds to comprehensively evaluate model performance and convergence behavior. To ensure effective convergence during training, we decreased the learning rate by 0.1 at both the halfway mark and three-quarters of the federated learning rounds.


# Acknowledgements

This codebase was adapted from https://doc.fedml.ai/ and https://github.com/Lee-Gihun/FedNTD/.
