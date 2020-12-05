import numpy as np
import time

import os
os.environ['keras'] = 'keras'

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import dataScrub as ds
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score

fake_data_loc = "Fake.csv"
true_data_loc = "True.csv"

fake_data_set, true_data_set = ds.getData(fake_data_loc, true_data_loc)

#set the traget labels
fake_data_set['isReal'] = 0
true_data_set['isReal'] = 1

#print(fake_data_set.head())

#Merging the 2 datasets
data_set = pd.concat([fake_data_set, true_data_set])
data_set = data_set.sample(frac=1).reset_index(drop = True) #shuffle

data_set['text'] = data_set['title'] + ' ' + data_set['text']

data_set = data_set.drop(['title','subject','date'],axis=1) # drop columns
#print(data_set.head())

#sns.countplot(data_set['isReal'])
#plt.show()
#print(data_set['text'][0])

start = time.time()
# clean training set with natura langauge proccessing
data_set['text'] = data_set['text'].apply(ds.scrubData)
#data_set['text'] = data_set['text'].split(' ')

print('Scrubbing time: ', round(time.time() - start, 2), 's')


'''
from tensorflow.keras.preprocessing import text, sequence
max_features = 1000
maxlen = 300
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(data_set['text'])
print(tokenizer.word_index)
exit()
tokenized_train = tokenizer.texts_to_sequences(data_set['text'])
data_set['text'] = sequence.pad_sequences(tokenized_train, dtype='float64')
'''
max_features = 100
#data = ds.TFIDF_vectorize(data_set['text'], max_features).tolist()
data_set['text'] = pd.Series(ds.TFIDF_vectorize(data_set['text'], max_features).tolist())

print(data_set['text'][0])

exit()
