GPU=3
IMGLR=0.1
for IPC in 10 20 30 50
do
	for OL in 500 1000 2000 3000
	do
	python ./main_feddf.py --gpu ${GPU} --client_num_in_total 20 --client_num_per_round 8 \
	--batch_size 64 --dataset cifar10 --data_dir /input/dataset/cifar10 \
	--model resnet8_no_bn --partition_method hetero --partition_alpha 0.01 --comm_round 100 \
	 --epochs 10 --lr 0.1 --client_optimizer sgd --frequency_of_the_test 1 --ci 1 \
	--valid_ratio 0.1 --unlabeled_data_dir /input/dataset/cifar100 \
	--server_steps 0 --unlabeled_dataset cifar100 --unlabeled_batch_size 128 \
	 --pname con-data-hp --condense --image_per_class ${IPC} --image_lr ${IMGLR}  \
	--train_condense_server --con_rand --condense_server_steps 1000 --condense_patience_steps 500 --condense_train_type soft \
	--condense_init  --init_outer_loops ${OL} --condense_batch_size 16 --condense_lr 0.001 --condense_optimizer adam \
	--coninit_save
	done 
done