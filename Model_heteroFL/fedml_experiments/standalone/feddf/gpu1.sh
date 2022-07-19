ALPHA=0.01
for SEED in 0 1 2
do
	for PART_NUM in 2 
	do
		for EPOCH in 5 10
		do
		python ./main_feddf.py --gpu 1 --client_num_in_total 20 --client_num_per_round ${PART_NUM} \
		--batch_size 64 --dataset cifar10 --data_dir /input/dataset/cifar10 \
		--model resnet8 --partition_method hetero --partition_alpha ${ALPHA} --comm_round 100 \
		 --epochs ${EPOCH} --lr 0.1 --client_optimizer sgd --frequency_of_the_test 1 --ci 1 \
		--valid_ratio 0.1 --unlabeled_data_dir /input/dataset/cifar100 --unlabeled_dataset cifar100 --unlabeled_batch_size 128 \
		--server_steps 1000 --server_patience_steps 500 --fedmix_server \
		--num_mixed_data_per_client 70 --seed ${SEED} --pname fedmix
		done
	done
done

