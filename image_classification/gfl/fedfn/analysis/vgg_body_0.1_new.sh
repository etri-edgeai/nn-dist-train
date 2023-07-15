python main_fed_setting3.py --fl_alg FedAvg --dataset cifar10 --scheduler multistep --lr_decay 0.1 --model vgg --hetero_option shard --shard_per_user 2 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 20 --local_bs 50 --local_upt_part body --momentum 0.90 --wd 1e-5 --gpu 6

python main_fed_setting3.py --fl_alg FedAvg --dataset cifar10 --scheduler multistep --lr_decay 0.1 --model vgg --hetero_option shard --shard_per_user 2 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 15 --local_bs 50 --local_upt_part body --momentum 0.90 --wd 1e-5 --gpu 6

python main_fed_setting3.py --fl_alg FedAvg --dataset cifar10 --scheduler multistep --lr_decay 0.1 --model vgg --hetero_option shard --shard_per_user 2 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --momentum 0.90 --wd 1e-5 --gpu 6

python main_fed_setting3.py --fl_alg FedAvg --dataset cifar10 --scheduler multistep --lr_decay 0.1 --model vgg --hetero_option shard --shard_per_user 2 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 --local_upt_part body --momentum 0.90 --wd 1e-5 --gpu 6

python main_fed_setting3.py --fl_alg FedAvg --dataset cifar10 --scheduler multistep --lr_decay 0.1 --model vgg --hetero_option shard --shard_per_user 2 --epochs 320 --lr 0.1 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --momentum 0.90 --wd 1e-5 --gpu 6