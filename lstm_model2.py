import numpy as np
import time
import pandas as pd
import dataScrub as ds

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import text, sequence

data_set = pd.read_csv('data_clean.csv')

#print(data_set.head())
data_train = data_set['text'][0:2000].astype(str)
data_labels = data_set['isReal'][0:2000].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(data_train, data_labels, test_size=0.25)

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


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau

batch_size = 64
epochs = 10
embed_size = 100

# Now lets build the model
model = Sequential() 

model.add(Embedding(max_features, output_dim=embed_size, input_length=maxlen, trainable=False)) #Embedding Layer

#model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2))) #Bi-directional LSTM
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#Dense layer
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid')) # binary classification (0\1)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, batch_size = batch_size , validation_data = (X_test, y_test) , epochs = epochs)

print("Accuracy of the model on Training Data is - " , model.evaluate(X_train, y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test, y_test)[1]*100 , "%")

#sample: 1000 epoch: 10 test auc= 82%
#sample: 2000 epoch: 20 test auc= 91%