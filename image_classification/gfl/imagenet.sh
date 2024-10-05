#MOON

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/moon.json --partition_method lda --partition_alpha 0.1  --device cuda:0 --seed 1 --lr 0.1 

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/moon.json --partition_method lda --partition_alpha 0.1  --device cuda:0 --seed 10 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/moon.json --partition_method lda --partition_alpha 0.1  --device cuda:0 --seed 100 --lr 0.1




OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/moon.json --partition_s 20 --device cuda:1 --seed 1 --lr 0.1 

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/moon.json --partition_s 20 --device cuda:1 --seed 10 --lr 0.1 

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/moon.json --partition_s 20 --device cuda:1 --seed 100 --lr 0.1 




#feddr

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_method lda --partition_alpha 0.1  --device cuda:0 --seed 1 --lr 1.0

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_method lda --partition_alpha 0.1  --device cuda:0 --seed 10 --lr 1.0

# OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_method lda --partition_alpha 0.1  --device cuda:0 --seed 100 --lr 1.0




OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_s 20 --device cuda:1 --seed 1 --lr 1.0 

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_s 20 --device cuda:1 --seed 10 --lr 1.0

# OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_s 20 --device cuda:1 --seed 100 --lr 1.0 





#feetf

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedetf.json --partition_method lda --partition_alpha 0.1  --device cuda:3 --seed 1 --lr 1.0

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedetf.json --partition_method lda --partition_alpha 0.1  --device cuda:3 --seed 10 --lr 1.0

# OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedetf.json --partition_method lda --partition_alpha 0.1  --device cuda:3 --seed 100 --lr 1.0



OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedetf.json --partition_s 20 --device cuda:3 --seed 1 --lr 1.0 

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedetf.json --partition_s 20 --device cuda:3 --seed 10 --lr 1.0

# OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedetf.json --partition_s 20 --device cuda:3 --seed 100 --lr 1.0 




#spherefed

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_method lda --partition_alpha 0.1  --device cuda:4 --seed 1 --lr 1.0

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_method lda --partition_alpha 0.1  --device cuda:4 --seed 10 --lr 1.0

# OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_method lda --partition_alpha 0.1  --device cuda:4 --seed 100 --lr 1.0




OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_s 20 --device cuda:4 --seed 1 --lr 1.0 

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_s 20 --device cuda:4 --seed 10 --lr 1.0

# OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/spherefed.json --partition_s 20 --device cuda:4 --seed 100 --lr 1.0 





#fedavg

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedavg.json --partition_method lda --partition_alpha 0.1  --device cuda:5 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedavg.json --partition_method lda --partition_alpha 0.1  --device cuda:5 --seed 10 --lr 0.1

# OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedavg.json --partition_method lda --partition_alpha 0.1  --device cuda:5 --seed 100 --lr 0.1




OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedavg.json --partition_s 20 --device cuda:5 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedavg.json --partition_s 20 --device cuda:5 --seed 10 --lr 0.1

# OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedavg.json --partition_s 20 --device cuda:5 --seed 100 --lr 0.1 




#fedbabu

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedbabu.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedbabu.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 10 --lr 0.1

# OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedbabu.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 100 --lr 0.1




OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedbabu.json --partition_s 20 --device cuda:6 --seed 1 --lr 0.1 

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedbabu.json --partition_s 20 --device cuda:6 --seed 10 --lr 0.1

# OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedbabu.json --partition_s 20 --device cuda:6 --seed 100 --lr 0.1 






#fedprox

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedprox.json --partition_method lda --partition_alpha 0.1  --device cuda:3 --seed 1 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedprox.json --partition_method lda --partition_alpha 0.1  --device cuda:4 --seed 10 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedprox.json --partition_method lda --partition_alpha 0.1  --device cuda:5 --seed 100 --lr 0.1



OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedprox.json --partition_s 20 --device cuda:1 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedprox.json --partition_s 20 --device cuda:4 --seed 10 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedprox.json --partition_s 20 --device cuda:4 --seed 100 --lr 0.1

































#scaffold

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/scaffold.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 1 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/scaffold.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 10 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/scaffold.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 100 --lr 0.1



OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/scaffold.json --partition_s 20 --device cuda:7 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/scaffold.json --partition_s 20 --device cuda:7 --seed 10 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/scaffold.json --partition_s 20 --device cuda:7 --seed 100 --lr 0.1



#fedgela

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedgela.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 1 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedgela.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 10 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedgela.json --partition_method lda --partition_alpha 0.1  --device cuda:6 --seed 100 --lr 0.1



OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedgela.json --partition_s 20 --device cuda:0 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedgela.json --partition_s 20 --device cuda:7 --seed 10 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedgela.json --partition_s 20 --device cuda:0 --seed 100 --lr 0.1



#fedsol

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedsol.json --partition_method lda --partition_alpha 0.1  --device cuda:2 --seed 1 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedsol.json --partition_method lda --partition_alpha 0.1  --device cuda:2 --seed 10 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedsol.json --partition_method lda --partition_alpha 0.1  --device cuda:2 --seed 100 --lr 0.1



OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedsol.json --partition_s 20 --device cuda:3 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedsol.json --partition_s 20 --device cuda:3 --seed 10 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedsol.json --partition_s 20 --device cuda:3 --seed 100 --lr 0.1



#fedntd

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedntd.json --partition_method lda --partition_alpha 0.1  --device cuda:0 --seed 1 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedntd.json --partition_method lda --partition_alpha 0.1  --device cuda:2 --seed 10 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedntd.json --partition_method lda --partition_alpha 0.1  --device cuda:2 --seed 100 --lr 0.1



OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedntd.json --partition_s 20 --device cuda:3 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedntd.json --partition_s 20 --device cuda:0 --seed 10 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedntd.json --partition_s 20 --device cuda:0 --seed 100 --lr 0.1







#fedexp

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedexp.json --partition_method lda --partition_alpha 0.1  --device cuda:4 --seed 1 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedexp.json --partition_method lda --partition_alpha 0.1  --device cuda:4 --seed 10 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedexp.json --partition_method lda --partition_alpha 0.1  --device cuda:5 --seed 100 --lr 0.1



OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedexp.json --partition_s 20 --device cuda:5 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedexp.json --partition_s 20 --device cuda:5 --seed 10 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedexp.json --partition_s 20 --device cuda:5 --seed 100 --lr 0.1
















OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedgela.json --partition_method lda --partition_alpha 0.1  --device cuda:7 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedprox.json --partition_method lda --partition_alpha 0.1  --device cuda:7 --seed 1 --lr 0.1


OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedntd.json --partition_method lda --partition_alpha 0.1  --device cuda:7 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedexp.json --partition_method lda --partition_alpha 0.1  --device cuda:7 --seed 1 --lr 0.1

OMP_NUM_THREADS=1 python main.py --config_path ./config/imagenet/fedsol.json --partition_method lda --partition_alpha 0.1  --device cuda:7 --seed 1 --lr 0.1



