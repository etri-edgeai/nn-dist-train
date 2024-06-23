
#CIFAR100

python main.py --config_path ./config/cifar100/spherefed.json --partition_s 10 --device cuda:7 --seed 100
python main.py --config_path ./config/cifar100/spherefed.json --partition_s 20 --device cuda:7 --seed 100
python main.py --config_path ./config/cifar100/spherefed.json --partition_s 50 --device cuda:7 --seed 100
python main.py --config_path ./config/cifar100/spherefed.json --partition_s 100 --device cuda:7 --seed 100
