#CIFAR100


python main.py --config_path ./config/cifar100/fedexp.json --partition_s 10 --device cuda:4 --seed 100
python main.py --config_path ./config/cifar100/fedexp.json --partition_s 20 --device cuda:4 --seed 100
python main.py --config_path ./config/cifar100/fedexp.json --partition_s 50 --device cuda:4 --seed 100
python main.py --config_path ./config/cifar100/fedexp.json --partition_s 100 --device cuda:4 --seed 100