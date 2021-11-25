if [ -f ./checkpoint/client_ckpt.pth ]; then 
    rm './checkpoint/client_ckpt.pth'
fi

if [ -f './optimizer/optimizer.pth' ]; then 
    rm './optimizer/optimizer.pth'
fi

rounds="10"

for r in $(seq 1 1 $rounds)
do
    python3 train_client.py --config_path ./config/3layer_fedavg_cifar10.py
    # python3 train_client.py --config_path ./config/3layer_fedlsd_cifar10.py
    # python3 train_client.py --config_path ./config/3layer_fedls-ntd_cifar10.py

    python3 socket_client.py
done
