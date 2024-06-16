#CIFAR10

python main.py --config_path ./config/cifar10/feddr.json --partition_method lda --partition_alpha 0.1  --device cuda:1 --seed 100
python main.py --config_path ./config/cifar10/feddr.json --partition_method lda --partition_alpha 0.2  --device cuda:1 --seed 100
python main.py --config_path ./config/cifar10/feddr.json --partition_method lda --partition_alpha 0.3  --device cuda:1 --seed 100



#CIFAR100

python main.py --config_path ./config/cifar100/feddr.json --partition_method lda --partition_alpha 0.05  --device cuda:1 --seed 100
python main.py --config_path ./config/cifar100/feddr.json --partition_method lda --partition_alpha 0.1  --device cuda:1 --seed 100
python main.py --config_path ./config/cifar100/feddr.json --partition_method lda --partition_alpha 0.2  --device cuda:1 --seed 100
python main.py --config_path ./config/cifar100/feddr.json --partition_method lda --partition_alpha 0.3  --device cuda:1 --seed 100


