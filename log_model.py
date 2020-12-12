import numpy as np
import time
import pandas as pd
import dataScrub as ds

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from tensorflow.keras.preprocessing import text, sequence

data_set = pd.read_csv('data_clean.csv')

#data_set['text'] = data_set['text'].astype(str).apply(ds.removeTrump)
#data_set['text'] = data_set['text'].astype(str).apply(ds.removeSaid)

#fake_data = data_set[data_set.isReal == 0]['text'].astype(str).apply(ds.removeTrump)
#true_data = data_set[data_set.isReal == 1]['text']
#data_set[data_set.isReal == 1]['text'] = data_set[data_set.isReal == 1]['text'].astype(str).apply(ds.removeSaid)
#fake_change = data_set[data_set.isReal == 0]['text'].replace('trump', '')
#true_change = data_set[data_set.isReal == 1]['text'].replace('said', '')

#print(data_set.head())
data_train = data_set['text'][0:10000].to_numpy().astype('U')
data_labels = data_set['isReal'][0:10000].to_numpy().astype('U')
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
vectorizer = CountVectorizer(max_features=10)
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)
print('Count Vec time: ', round(time.time() - start, 2), 's')

start = time.time()
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print('Model Fit time: ', round(time.time() - start, 2), 's')
score = classifier.score(X_test, y_test)
print(score)

#classification report
from sklearn.metrics import classification_report,confusion_matrix
y_pred = classifier.predict(X_test)
print(classification_report(y_test,y_pred))
print('Confusion matix:\n',confusion_matrix(y_test,y_pred))