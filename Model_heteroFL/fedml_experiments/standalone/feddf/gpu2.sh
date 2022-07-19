ALPHA=0.01
SEED=1
for CLIENT_NUM in 2 8
do
	for EPOCH in 5 10
	do
		python ./main_feddf.py --gpu 2 --client_num_in_total 20 --client_num_per_round ${CLIENT_NUM} \
		--batch_size 64 --dataset cifar10 --data_dir /input/dataset/cifar10 \
		--model resnet8_no_bn --partition_method hetero --partition_alpha ${ALPHA} --comm_round 100 \
		 --epochs ${EPOCH} --lr 0.1 --client_optimizer sgd --frequency_of_the_test 1 --ci 1 \
		--valid_ratio 0.1 --unlabeled_data_dir /input/dataset/cifar100 \
		--server_steps 0 --unlabeled_dataset cifar100 --unlabeled_batch_size 128 \
		 --pname prox-con-re --condense --image_per_class 20 --image_lr 1  \
		--train_condense_server --con_rand --condense_server_steps 1000 --condense_patience_steps 500 --condense_train_type soft \
		--condense_init  --init_outer_loops 10000 --condense_batch_size 16 --condense_lr 0.001  \
		--coninit_load --condense_reinit_model --lambda_fedprox 0.01 --seed ${SEED}


		python ./main_feddf.py --gpu 2 --client_num_in_total 20 --client_num_per_round ${CLIENT_NUM} \
		--batch_size 64 --dataset cifar10 --data_dir /input/dataset/cifar10 \
		--model resnet8_no_bn --partition_method hetero --partition_alpha ${ALPHA} --comm_round 100 \
		 --epochs  ${EPOCH} --lr 0.1 --client_optimizer sgd --frequency_of_the_test 1 --ci 1 \
		--valid_ratio 0.1 --unlabeled_data_dir /input/dataset/cifar100 --server_steps 0 \
		--lambda_fedprox 0.01 --pname prox --seed ${SEED}
	done
done