from datasets import load_dataset
import pandas as pd
import os

os.makedirs("MNLI")

mnli_train = load_dataset('mnli', split='train', cache_dir='') # data 폴더 안에 따로 만드는 게 좋을 듯
mnli_test = load_dataset('mnli', split='validation_matched', cache_dir='')

gov_train=[]
for i in mnli_train:
    if i['genre']=='government':
        gov_train.append(i['input_text']+'\t'+i['target_text'])
    
telephone_train=[]
for i in mnli_train:
    if i['genre']=='telephone':
        telephone_train.append(i['input_text']+'\t'+i['target_text'])
    
fiction_train=[]
for i in mnli_train:
    if i['genre']=='fiction':
        fiction_train.append(i['input_text']+'\t'+i['target_text'])
    
travel_train=[]
for i in mnli_train:
    if i['genre']=='travel':
        travel_train.append(i['input_text']+'\t'+i['target_text'])
    
slate_train=[]
for i in mnli_train:
    if i['genre']=='slate':
        slate_train.append(i['input_text']+'\t'+i['target_text'])
        
gov_test=[]
for i in mnli_test:
    if i['genre']=='government':
        gov_test.append(i['input_text']+'\t'+i['target_text'])
    
telephone_test=[]
for i in mnli_test:
    if i['genre']=='telephone':
        telephone_test.append(i['input_text']+'\t'+i['target_text'])
    
fiction_test=[]
for i in mnli_test:
    if i['genre']=='fiction':
        fiction_test.append(i['input_text']+'\t'+i['target_text'])
    
travel_test=[]
for i in mnli_test:
    if i['genre']=='travel':
        travel_test.append(i['input_text']+'\t'+i['target_text'])
    
slate_test=[]
for i in mnli_test:
    if i['genre']=='slate':
        slate_test.append(i['input_text']+'\t'+i['target_text'])
        
train_gov=pd.DataFrame(gov_train)
train_telephone=pd.DataFrame(telephone_train)
train_fiction=pd.DataFrame(fiction_train)
train_travel=pd.DataFrame(travel_train)
train_slate=pd.DataFrame(slate_train)

train_gov.to_csv("./MNLI/gov.train", index=None)
train_telephone.to_csv("./MNLI/tel.train", index=None)
train_fiction.to_csv("./MNLI/fic.train", index=None)
train_travel.to_csv("./MNLI/travel.train", index=None)
train_slate.to_csv("./MNLI/slate.train", index=None)


test_gov=pd.DataFrame(gov_test)
test_telephone=pd.DataFrame(telephone_test)
test_fiction=pd.DataFrame(fiction_test)
test_travel=pd.DataFrame(travel_test)
test_slate=pd.DataFrame(slate_test)

test_gov.to_csv("./MNLI/gov.test", index=None)
test_telephone.to_csv("./MNLI/tel.test", index=None)
test_fiction.to_csv("./MNLI/fic.test", index=None)
test_travel.to_csv("./MNLI/travel.test", index=None)
test_slate.to_csv("./MNLI/slate.test", index=None)
