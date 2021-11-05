# wandb
group = "Reproduce"
name = "Res8_FesLSD"
tags = []

device = "cuda:0"

configdict = {
    # Experiment information
    "exp_info": {
        "project_name": "reproduce",
        "name": name,
        "group": group,
        "tags": tags,
        "notes": "",
    },
    # Federated Learning setups
    "fed_setups": {
        "model": {
            "name": "res8",
            "params": {"use_bias": True},
        },  # name: cifarcnn, res8, vgg11
        "server_params": {
            "n_rounds": 300,
            "n_clients": 100,
            "sample_ratio": 0.1,
            "agg_alg": "fedavg",
            "server_location": "cpu",
            "device": device,
        },
        "local_params": {
            "local_ep": 5,
            "local_bs": 50,
            "global_loss": "none", 
            "global_alpha": 0.0,
        }, # global_loss: none, proximal
    },
    # Data setups
    "data_setups": {
        "dataset_params": {
            "root": "./data",
            "dataset": "cifar10",
        },
        "distribute_params": {
            "alg": "fedavg",
            "max_class_num": 2,
        },  # alg: iid, fedavg(with max_class_num option), fedma(with dir_alpha option)
    },
    # Evaluation setups
    "evaluation": {
        "in_inter_angle": {"enabled": False},
        "cka": {"enabled": False, "period": 1, "mode": "ALL"},
        "global_analysis": {"enabled": False},
        "device": device,
    },
    # Training setups
    "criterion": {
        "params": {
            "mode": "LSD",
            "beta": 0.3,
            "temp": 3,
            "num_classes": 10,
        } # mode: CE, LSD, LS-NTD, LSD_NTD & lam: only for LSD_NTD
    },
    "optimizer": {"params": {"lr": 0.03, "momentum": 0.9, "weight_decay": 0}},
    "scheduler": {
        "enabled": True,
        "type": "step",  # cosine, multistep, step
        "params": {"gamma": 0.99, "step_size": 5},
    },
    "seed": 2021,
}
