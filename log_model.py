import numpy as np
import time
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

data_set = pd.read_csv('data_clean.csv')

#print(data_set.head())
data_train = data_set['text'][0:10000].to_numpy().astype('U')
data_labels = data_set['isReal'][0:10000].to_numpy().astype('U')
X_train, X_test, y_train, y_test = train_test_split(data_train, data_labels, test_size=0.25)

start = time.time()
vectorizer = CountVectorizer()
vectorizer.fit(X_train)
X_train = vectorizer.transform(X_train)
X_test  = vectorizer.transform(X_test)
print('Count Vec time: ', round(time.time() - start, 2), 's')

start = time.time()
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print('ModelFit time: ', round(time.time() - start, 2), 's')

score = classifier.score(X_test, y_test)
print(score)