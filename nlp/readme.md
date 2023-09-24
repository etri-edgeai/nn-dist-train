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
