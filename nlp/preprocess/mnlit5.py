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

#os.makedirs("rte")
os.makedirs("mnli")

ag_train = load_dataset('multi_nli', split='train', cache_dir='./download') # data 폴더 안에 따로 만드는 게 좋을 듯
ag_test = load_dataset('multi_nli', split='validation_matched', cache_dir='./download')


def add_eos_to_examples(example):
    answer_dict={0:'entailment', 1:'neutral', 2:'contradiction'}
    answer = answer_dict[example['label']]
    example['input_text'] = 'mnli hypothesis: %s  premise: %s </s>' % (example['hypothesis'], example['premise'])
    example['target_text'] = '%s </s>' % answer
    del example['promptID']
    del example['pairID']
    del example['premise']
    del example['hypothesis']
    del example['premise_binary_parse']
    del example['premise_parse']
    del example['hypothesis_binary_parse']
    del example['hypothesis_parse']
    del example['label']
    return example


mnli_train = ag_train.map(add_eos_to_examples)
mnli_test = ag_test.map(add_eos_to_examples)


gov_train=[]
for i in mnli_train:
    if i['genre']=='government':
        gov_train.append({
                    "input": i['input_text'], # combine news title with news body
                    "target": i['target_text'],
                })
    
telephone_train=[]
for i in mnli_train:
    if i['genre']=='telephone':
        telephone_train.append({
                    "input": i['input_text'], # combine news title with news body
                    "target": i['target_text'],
                })
    
fiction_train=[]
for i in mnli_train:
    if i['genre']=='fiction':
        fiction_train.append({
                    "input": i['input_text'], # combine news title with news body
                    "target": i['target_text'],
                })
    
travel_train=[]
for i in mnli_train:
    if i['genre']=='travel':
        travel_train.append({
                    "input": i['input_text'], # combine news title with news body
                    "target": i['target_text'],
                })
    
slate_train=[]
for i in mnli_train:
    if i['genre']=='slate':
        slate_train.append({
                    "input": i['input_text'], # combine news title with news body
                    "target": i['target_text'],
                })
        
gov_test=[]
for i in mnli_test:
    if i['genre']=='government':
        gov_test.append({
                    "input": i['input_text'], # combine news title with news body
                    "target": i['target_text'],
                })
    
telephone_test=[]
for i in mnli_test:
    if i['genre']=='telephone':
        telephone_test.append({
                    "input": i['input_text'], # combine news title with news body
                    "target": i['target_text'],
                })
    
fiction_test=[]
for i in mnli_test:
    if i['genre']=='fiction':
        fiction_test.append({
                    "input": i['input_text'], # combine news title with news body
                    "target": i['target_text'],
                })
    
travel_test=[]
for i in mnli_test:
    if i['genre']=='travel':
        travel_test.append({
                    "input": i['input_text'], # combine news title with news body
                    "target": i['target_text'],
                })
    
slate_test=[]
for i in mnli_test:
    if i['genre']=='slate':
        slate_test.append({
                    "input": i['input_text'], # combine news title with news body
                    "target": i['target_text'],
                })
        
train_gov=pd.DataFrame(gov_train)
train_telephone=pd.DataFrame(telephone_train)
train_fiction=pd.DataFrame(fiction_train)
train_travel=pd.DataFrame(travel_train)
train_slate=pd.DataFrame(slate_train)

train_gov.to_csv("./mnli/gov_train.csv", index=None)
train_telephone.to_csv("./mnli/tel_train.csv", index=None)
train_fiction.to_csv("./mnli/fic_train.csv", index=None)
train_travel.to_csv("./mnli/travel_train.csv", index=None)
train_slate.to_csv("./mnli/slate_train.csv", index=None)


test_gov=pd.DataFrame(gov_test)
test_telephone=pd.DataFrame(telephone_test)
test_fiction=pd.DataFrame(fiction_test)
test_travel=pd.DataFrame(travel_test)
test_slate=pd.DataFrame(slate_test)

test_gov.to_csv("./mnli/gov_test.csv", index=None)
test_telephone.to_csv("./mnli/tel_test.csv", index=None)
test_fiction.to_csv("./mnli/fic_test.csv", index=None)
test_travel.to_csv("./mnli/travel_test.csv", index=None)
test_slate.to_csv("./mnli/slate_test.csv", index=None)




