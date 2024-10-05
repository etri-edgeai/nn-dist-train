#CIFAR10

python main.py --config_path ./config/cifar10/moon.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 100
python main.py --config_path ./config/cifar10/moon.json --partition_method lda --partition_alpha 0.2  --device cuda:6 --seed 100
python main.py --config_path ./config/cifar10/moon.json --partition_method lda --partition_alpha 0.3  --device cuda:6 --seed 100

