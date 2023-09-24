from datasets import load_dataset
import pandas as pd
import os
import time
import glob
import json
import random
import numpy as np

np.random.seed(1)
random.seed(1)

os.makedirs("qnli")

ag_train = load_dataset('glue','qnli', split='train', cache_dir='./download') # data 폴더 안에 따로 만드는 게 좋을 듯
ag_test = load_dataset('glue','qnli', split='validation', cache_dir='./download')

train=[]
for i in ag_train:
    train.append({
                    "input": i['question']+i['sentence'], # combine news title with news body
                    "label": i['label'],
                })    
    
test=[]
for i in ag_test:
    test.append({
                    "input": i['question']+i['sentence'], # combine news title with news body
                    "label": i['label'],
                })

train=pd.DataFrame(train)
test=pd.DataFrame(test)
train1=train[:34914]
train2=train[34914:69828]
train3=train[69828:104742]

test1=test[:]
test2=test[:]
test3=test[:]
#test4=test[:]
            


train1.to_csv("./qnli/1_train.csv", index=None)
train2.to_csv("./qnli/2_train.csv", index=None)
train3.to_csv("./qnli/3_train.csv", index=None)
#train4.to_csv("./rte/4_train.csv", index=None)


test1.to_csv("./qnli/1_test.csv", index=None)
test2.to_csv("./qnli/2_test.csv", index=None)
test3.to_csv("./qnli/3_test.csv", index=None)
#test4.to_csv("./rte/4_test.csv", index=None)




