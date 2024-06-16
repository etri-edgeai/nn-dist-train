#CIFAR10

python main.py --config_path ./config/cifar10/fedexp.json --partition_method lda --partition_alpha 0.1  --device cuda:4 --seed 100
python main.py --config_path ./config/cifar10/fedexp.json --partition_method lda --partition_alpha 0.2  --device cuda:4 --seed 100
python main.py --config_path ./config/cifar10/fedexp.json --partition_method lda --partition_alpha 0.3  --device cuda:4 --seed 100
0




#CIFAR100

python main.py --config_path ./config/cifar100/fedexp.json --partition_method lda --partition_alpha 0.05  --device cuda:4 --seed 100
python main.py --config_path ./config/cifar100/fedexp.json --partition_method lda --partition_alpha 0.1  --device cuda:4 --seed 100
python main.py --config_path ./config/cifar100/fedexp.json --partition_method lda --partition_alpha 0.2  --device cuda:4 --seed 100
python main.py --config_path ./config/cifar100/fedexp.json --partition_method lda --partition_alpha 0.3  --device cuda:4 --seed 100


