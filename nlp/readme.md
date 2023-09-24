## Requirements
### Python modules
```
pip install -r requirements.txt
```

## Example to Run
### Preprocess datasets
```
python preprocess/qasc.py
python preprocess/rte.py
```

## running standard fine-tuning code
```
python main_lm.py --data cola --model bert-large-uncased --n_cpu 1.0 --n_gpus 2 --frac_fit 1.0 --epoch 1 --batch_size 16 --batch_accum 2 --lang_mix 0.99 --centralized --n_iterations 5 --lr 1e-5 --seed 5 > ./log/bert-large/cola/central5.txt
```
## running fl code 
```
python main_lm.py --data cola --model bert-large-uncased --n_cpu 1.0 --n_gpus 2 --frac_fit 1.0 --epoch 2 --batch_size 16 --batch_accum 2 --lang_mix 0.99 --n_iterations 5 --lr 1e-5 --seed 5 > ./log/bert-large/cola/iid5.txt
```

## explain of each api

--data : which data you want to run

--model : which model you'll use

--n_cpu: how many cpu cores are allocated for each client

--n_gpu: how many gpus are used to run this entire code

--frac_fit: client selection ratio


