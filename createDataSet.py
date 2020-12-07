import numpy as np
import time
import pandas as pd

import dataScrub as ds

fake_data_loc = "Fake.csv"
true_data_loc = "True.csv"

#extract fake and true data sets
fake_data_set = pd.read_csv(fake_data_loc)
true_data_set = pd.read_csv(true_data_loc)

#set the traget labels
fake_data_set['isReal'] = 0
true_data_set['isReal'] = 1

#Merging the 2 datasets
data_set = pd.concat([fake_data_set, true_data_set])
data_set = data_set.sample(frac=1).reset_index(drop = True) #shuffle
data_set = data_set.sample(frac=1).reset_index(drop = True) #shuffle

data_set['text'] = data_set['title'] + ' ' + data_set['text'] #combine the tittle and text together

data_set = data_set.drop(['title','subject','date'],axis=1) # drop columns

data_set_clean = data_set.copy()
#data_set_clean['isReal'] = pd.Series(data_set['isReal'])
print(data_set_clean.head())

data_set.to_csv('data.csv', index = False)

start = time.time()
data_set_clean['text'] = data_set_clean['text'].apply(ds.scrubData)
print('Scrubbing time: ', round(time.time() - start, 2), 's')

print(data_set_clean.head())

data_set_clean.to_csv('data_clean.csv', index = False)
