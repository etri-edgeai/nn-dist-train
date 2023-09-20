#LDA

python main_fed.py --fl_alg FedAvg --dataset cifar100 --lr_decay 0.1 --model mobile --num_classes 100 --hetero_option lda --alpha 0.1 --epochs 320 --lr 0.5 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 --local_upt_part full --momentum 0.90 --wd 1e-5 --gpu 0 --fn

python main_fed.py --fl_alg FedAvg --dataset cifar100 --lr_decay 0.1 --model mobile --num_classes 100 --hetero_option lda --alpha 0.5 --epochs 320 --lr 0.5 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 --local_upt_part full --momentum 0.90 --wd 1e-5 --gpu 0 --fn

python main_fed.py --fl_alg FedAvg --dataset cifar100 --lr_decay 0.1 --model mobile --num_classes 100 --hetero_option lda --alpha 1.0 --epochs 320 --lr 0.5 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 --local_upt_part full --momentum 0.90 --wd 1e-5 --gpu 0 --fn


#Shard

python main_fed.py --fl_alg FedAvg --dataset cifar100 --lr_decay 0.1 --model mobile --num_classes 100 --hetero_option shard --shard_per_user 10 --epochs 320 --lr 0.5 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 --local_upt_part full --momentum 0.90 --wd 1e-5 --gpu 0 --fn

python main_fed.py --fl_alg FedAvg --dataset cifar100 --lr_decay 0.1 --model mobile --num_classes 100 --hetero_option shard --shard_per_user 50 --epochs 320 --lr 0.5 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 --local_upt_part full --momentum 0.90 --wd 1e-5 --gpu 0 --fn

python main_fed.py --fl_alg FedAvg --dataset cifar100 --lr_decay 0.1 --model mobile --num_classes 100 --hetero_option shard --shard_per_user 100 --epochs 320 --lr 0.5 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 50 --local_upt_part full --momentum 0.90 --wd 1e-5 --gpu 0 --fn