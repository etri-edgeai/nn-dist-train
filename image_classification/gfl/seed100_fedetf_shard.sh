#CIFAR10
python main.py --config_path ./config/cifar10/fedetf.json --partition_method sharding --partition_s 2 --device cuda:3 --seed 100
python main.py --config_path ./config/cifar10/fedetf.json --partition_method sharding --partition_s 5 --device cuda:3 --seed 100
python main.py --config_path ./config/cifar10/fedetf.json --partition_method sharding --partition_s 10 --device cuda:3 --seed 100




#CIFAR100
python main.py --config_path ./config/cifar100/fedetf.json --partition_s 10 --device cuda:3 --seed 100
python main.py --config_path ./config/cifar100/fedetf.json --partition_s 20 --device cuda:3 --seed 100
python main.py --config_path ./config/cifar100/fedetf.json --partition_s 50 --device cuda:3 --seed 100
python main.py --config_path ./config/cifar100/fedetf.json --partition_s 100 --device cuda:3 --seed 100
