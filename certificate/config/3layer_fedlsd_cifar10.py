# wandb
group = "Reproduce"
name = "3layer_FesLSD"
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
        "local_params": {
            "local_ep": 1,
            "local_bs": 128,
            "global_loss": "none", 
            "global_alpha": 0.0,
        }, # global_loss: none, proximal
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
    "optimizer": {"params": {"lr": 0.05, "momentum": 0.9, "weight_decay": 0}},
    "scheduler": {
        "enabled": True,
        "type": "step",  # cosine, multistep, step
        "params": {"gamma": 0.99, "step_size": 5},
    },
    "seed": 2021,
}
