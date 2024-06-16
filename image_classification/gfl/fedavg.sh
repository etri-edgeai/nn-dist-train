python main.py --config_path ./config/cifar100/fedavg.json --partition_s 10 --device cuda:5 --seed 100
python main.py --config_path ./config/cifar100/fedavg.json --partition_s 20 --device cuda:5 --seed 100
python main.py --config_path ./config/cifar100/fedavg.json --partition_s 50 --device cuda:5 --seed 100
python main.py --config_path ./config/cifar100/fedavg.json --partition_s 100 --device cuda:5 --seed 100


#LDA
#CIFAR100

python main.py --config_path ./config/cifar100/fedavg.json --partition_method lda --partition_alpha 0.05  --device cuda:6 --seed 100
python main.py --config_path ./config/cifar100/fedavg.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 100
python main.py --config_path ./config/cifar100/fedavg.json --partition_method lda --partition_alpha 0.2  --device cuda:6 --seed 100
python main.py --config_path ./config/cifar100/fedavg.json --partition_method lda --partition_alpha 0.3  --device cuda:6 --seed 100

