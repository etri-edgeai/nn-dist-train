#CIFAR10

python main.py --config_path ./config/cifar10/spherefed.json --partition_method lda --partition_alpha 0.1  --device cuda:7 --seed 10
python main.py --config_path ./config/cifar10/spherefed.json --partition_method lda --partition_alpha 0.2  --device cuda:7 --seed 10
python main.py --config_path ./config/cifar10/spherefed.json --partition_method lda --partition_alpha 0.3  --device cuda:7 --seed 10



#CIFAR100

python main.py --config_path ./config/cifar100/spherefed.json --partition_method lda --partition_alpha 0.05  --device cuda:7 --seed 10
python main.py --config_path ./config/cifar100/spherefed.json --partition_method lda --partition_alpha 0.1  --device cuda:7 --seed 10
python main.py --config_path ./config/cifar100/spherefed.json --partition_method lda --partition_alpha 0.2  --device cuda:7 --seed 10
python main.py --config_path ./config/cifar100/spherefed.json --partition_method lda --partition_alpha 0.3  --device cuda:7 --seed 10


