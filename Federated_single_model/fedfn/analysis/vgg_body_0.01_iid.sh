python main_fed.py --fl_alg FedAvg --dataset cifar10 --scheduler multistep --lr_decay 0.1 --model vgg --epochs 320 --lr 0.01 --num_users 100 --frac 0.1 --local_ep 20 --local_bs 50 --local_upt_part body --momentum 0.90 --wd 1e-5 --gpu 7 --iid

python main_fed.py --fl_alg FedAvg --dataset cifar10 --scheduler multistep --lr_decay 0.1 --model vgg --epochs 320 --lr 0.01 --num_users 100 --frac 0.1 --local_ep 15 --local_bs 50 --local_upt_part body --momentum 0.90 --wd 1e-5 --gpu 7 --iid

python main_fed.py --fl_alg FedAvg --dataset cifar10 --scheduler multistep --lr_decay 0.1 --model vgg --epochs 320 --lr 0.01 --num_users 100 --frac 0.1 --local_ep 10 --local_bs 50 --local_upt_part body --momentum 0.90 --wd 1e-5 --gpu 7 --iid

python main_fed.py --fl_alg FedAvg --dataset cifar10 --scheduler multistep --lr_decay 0.1 --model vgg --epochs 320 --lr 0.01 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 --local_upt_part body --momentum 0.90 --wd 1e-5 --gpu 7 --iid

python main_fed.py --fl_alg FedAvg --dataset cifar10 --scheduler multistep --lr_decay 0.1 --model vgg --epochs 320 --lr 0.01 --num_users 100 --frac 0.1 --local_ep 1 --local_bs 50 --local_upt_part body --momentum 0.90 --wd 1e-5 --gpu 7 --iid