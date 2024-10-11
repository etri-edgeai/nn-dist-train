# Federated Learning for Re-ID task

Personal re-identification is an important computer vision task, but its development is constrained by the increasing privacy concerns. Federated learning is a privacy-preserving machine learning technique that learns a shared model across decentralized clients. In this work, we implement federated learning to person re-identification (**FedReID**) and optimize its performance affected by **statistical heterogeneity** in the real-world scenario. 

This repository is designed mainly for research, and pytorch implementation of federated learning algorithm on personal re-id task. 

Implemented algorithm are as follows: 

- **FedPAV** ([Paper Link](https://dl.acm.org/doi/pdf/10.1145/3531013), presented at **ACMMM 2020**),
- **FedDKD (Ours)** ([Paper Link](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003117176), presented at **KIISE 2022**),
- **FedCON (Ours)** ([Paper Link](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003117176), presented at **KIISE 2024**).
- **FedCON+ (Ours)** (Proposed in 2024).



## Prerequisite
* Install the libraries listed in requirements.txt
    ```
    pip install -r requirements.txt
    ```
## Dataset 
* Get the pre-processed dataset which is done by cap-ntu FedReID team (https://github.com/cap-ntu/FedReID). Dataset consist of 9 popular person ReID datasets

## Arguments for running code 
- `--gpu_ids`: Specify GPU IDs for training (e.g., `0`, `0,1,2`, `0,2`).
- `--model_name`: Set the model architecture (e.g., `ft_ResNet50`).
- `--project_dir`: Define the project directory path (default: `./`).
- `--data_dir`: Specify the training data directory path (default: `data`).
- `--datasets`: List of datasets to be used, separated by commas (e.g., `Market`, `DukeMTMC-reID`, `cuhk01`, `MSMT17`).
- `--train_all`: Flag to use all available training data (default: `True`).
- `--stride`: Set the stride for training (default: `2`).
- `--lr`: Learning rate for model training (default: `0.05`).
- `--drop_rate`: Dropout rate for the model (default: `0.5`).
- `--local_epoch`: Number of local epochs in federated setting (default: `2`).
- `--batch_size`: Batch size for local model updates (default: `32`).
- `--num_of_clients`: Number of federated clients (default: `9`).
- `--erasing_p`: Probability of random erasing during training (default: `0`, in range `[0, 1]`).
- `--color_jitter`: Flag to enable color jitter during training.
- `--which_epoch`: Select the specific epoch to test (e.g., `0`, `1`, `2`, `3`... or `last`).
- `--multi`: Flag to use multiple query options during evaluation.
- `--multiple_scale`: Flag for applying multiple scaling during testing (default: `1`, e.g., `1,1.1,1.2`).
- `--test_dir`: Directory path for test data (default: `all`).
- `--cdw`: Enable cosine distance weighting for model aggregation (default: `False`).
- `--kd`: Flag for enabling knowledge distillation during training (default: `False`).
- `--regularization`: Enable regularization during knowledge distillation (default: `False`).
- `--tau`: NTD loss hyperparameter controlling the weight of loss term (default: `3`).
- `--beta`: NTD loss hyperparameter affecting regularization (default: `1`).
- `--strategy`: The federated strategy to use (default: `fedpav`).

## Run the experiments
For running experiment with cross-silo setting (every client participate in every round), please use following code. 
* Run Federated Partial Averaging (FedPav): 
    ```
    python main.py
    ```
* Run Federated Not-True Distillation (FedDKD): 
    ```
    python main.py --strategy feddkd
    ```
* Run Model-contrastive federated learning (FedCON): 
    ```
    python main.py --strategy fedcon
    ```

* Run Model-contrastive federated learning (FedCON+): 
    ```
    python main.py --strategy fedcon_plus
    ```

## Result of implemented algorithm 

|Dataset    | fedpav(3) | feddkd(3) | fedcon(3) | fedcon+(3) |
|-----------|-----------|-----------|---------|---------------|
|MSMT17     | 0.6354    | 0.6419    | 0.6488  | 0.6594        |
|DUKE-MTMC  | 0.6171    | 0.6117    | 0.6041  | 0.6127        |
|Market-1501| 0.1271    | 0.1529    | 0.1407  | 0.1629        |
|CHUK03-NP  | 0.6101    | 0.6554    | 0.6224  | 0.6327        |
|PRID2011   | 0.282     | 0.3296    | 0.2697  | 0.2971        |
|CHUK01     | 0.3481    | 0.3291    | 0.3196  | 0.3513        |
|VIPeR      | 0.12      | 0.09      | 0.06    | 0.09          |
|3DPeS      | 0.5894    | 0.6301    | 0.626   | 0.6545        |
|iLIDS-VID  | 0.7449    | 0.7347    | 0.7653  | 0.8061        |

(3) indicates the number of local epoch (which is 3) when running algorithm. 
