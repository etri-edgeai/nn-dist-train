
# Install PyTorch (please visit pytorch.org to check your version according to your physical machines
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# Install MPI
conda install -c anaconda mpi4py

# Install Wandb
pip install --upgrade wandb

# Install other required package
conda install scikit-learn
conda install numpy
conda install h5py
conda install setproctitle
conda install networkx
pip install -r requirements.txt

# install the dataset
# 1. MNIST
cd ./data/MNIST
sh download_and_unzip.sh
cd ../../

# 2. FederatedEMNIST
cd ./data/FederatedEMNIST
sh download_federatedEMNIST.sh
cd ../../

# 3. shakespeare
cd ./data/shakespeare
sh download_shakespeare.sh
cd ../../


# 4. fed_shakespeare
cd ./data/fed_shakespeare
sh download_shakespeare.sh
cd ../../

# 5. fed_cifar100
cd ./data/fed_cifar100
sh download_fedcifar100.sh
cd ../../

# 6. stackoverflow
# cd ./data/stackoverflow
# sh download_stackoverflow.sh
# cd ../../

# 7. CIFAR10
cd ./data/cifar10
sh download_cifar10.sh
cd ../../

# 8. CIFAR100
cd ./data/cifar100
sh download_cifar100.sh
cd ../../

# 9. CINIC10
cd ./data/cinic10
sh download_cinic10.sh > cinic10_downloading_log.txt
cd ../../
