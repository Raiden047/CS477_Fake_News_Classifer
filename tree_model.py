import numpy as np
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing import text, sequence

data_set = pd.read_csv('data_clean.csv')

#print(data_set.head())
data_train = data_set['text'][0:20000].to_numpy().astype('U')
data_labels = data_set['isReal'][0:20000].to_numpy().astype('U')
X_train, X_test, y_train, y_test = train_test_split(data_train, data_labels, test_size=0.25)

'''
max_features = 1000
maxlen = 300
start = time.time()
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)

tokenized_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

tokenized_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
print('keras tokenize time: ', round(time.time() - start, 2), 's')

'''
start = time.time()
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)
print('Count Vec time: ', round(time.time() - start, 2), 's')

DTC = tree.DecisionTreeClassifier(max_features=1000)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(DTC, X_train, y_train, cv=5)
print(scores.mean())
#DTC.fit(X_train, y_train)
#print(DTC.score(X_test, y_test))