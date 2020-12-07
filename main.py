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

data_loc = "data.csv"

data_set = ds.getData(data_loc)

#print(data_set.head())

start = time.time()
# clean training set with natura langauge proccessing
data_set['text'] = data_set['text'].apply(ds.scrubData)
#data_set['text'] = data_set['text'].split(' ')

print('Scrubbing time: ', round(time.time() - start, 2), 's')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_set.text, data_set.isReal, random_state = 0)

from tensorflow.keras.preprocessing import text, sequence
max_features = 1000
maxlen = 300
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(x_train)

tokenized_train = tokenizer.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

tokenized_test = tokenizer.texts_to_sequences(x_test)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)

#max_features = 100
#data = ds.TFIDF_vectorize(data_set['text'], max_features).tolist()
#data_set['text'] = pd.Series(ds.TFIDF_vectorize(data_set['text'], max_features).tolist())

#print(x_train[0])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau

batch_size = 256
epochs = 10

# Now lets build the model
model = Sequential() 

model.add(Embedding(max_features, output_dim = 128)) #Embedding Layer

model.add(Bidirectional(LSTM(128))) #Bi-directional LSTM

#Dense layer
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid')) # binary classification (0\1)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['acc'])
model.summary()

y_train = np.asarray(y_train)

#learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)
# train the model
#history = model.fit(x_train, y_train, batch_size = batch_size , validation_data = (x_test, y_test) , epochs = epochs , callbacks = [learning_rate_reduction])
model.fit(x_train, y_train, batch_size= 64, validation_split = 0.1, epochs= 2)

pred = model.predict(x_test)
prediction = []
for i in range (len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(list(y_test), prediction)

print ("The model accuracy is :", accuracy)

#print("Accuracy of the model on Training Data is - " , model.evaluate(x_train,y_train)[1]*100 , "%")
#print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test,y_test)[1]*100 , "%")


exit()
