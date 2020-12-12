import numpy as np
import time
import pandas as pd
import dataScrub as ds

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import text, sequence

data_set = pd.read_csv('data_clean.csv')

#print(data_set.head())
data_train = data_set['text'][0:1000].astype(str)
data_labels = data_set['isReal'][0:1000].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(data_train, data_labels, test_size=0.25, random_state=42)


max_features = 10
maxlen = 500

start = time.time()
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)

tokenized_train = tokenizer.texts_to_sequences(X_train)
X_train = sequence.pad_sequences(tokenized_train, maxlen=maxlen)

tokenized_test = tokenizer.texts_to_sequences(X_test)
X_test = sequence.pad_sequences(tokenized_test, maxlen=maxlen)
print('keras tokenize time: ', round(time.time() - start, 2), 's')

#print(X_train[0])

EMBEDDING_FILE = 'twitter\glove.twitter.27B.100d.txt'

start = time.time()

def get_coefs(word, *arr): 
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding="utf8"))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
embed_size = all_embs.shape[1]

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
#change below line if computing normal stats is too slow
embedding_matrix = embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

print('glove emmbed time: ', round(time.time() - start, 2), 's')
#print(embedding_matrix[0])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau

batch_size = 64
epochs = 10
embed_size = 100

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience = 2, verbose=1, factor=0.5, min_lr=0.00001)

# Now lets build the model
model = Sequential() 

model.add(Embedding(max_features, output_dim=embed_size, weights=[embedding_matrix], input_length=maxlen, trainable=False)) #Embedding Layer
model.add(Dropout(0.3))
model.add(LSTM(128))
model.add(Dense(1, activation = 'sigmoid')) # binary classification

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics= ['acc'])
model.summary()

history = model.fit(X_train, y_train, batch_size = batch_size , validation_data = (X_test, y_test) , epochs = epochs)

print("Accuracy of the model on Training Data is - " , model.evaluate(X_train, y_train)[1]*100 , "%")
print("Accuracy of the model on Testing Data is - " , model.evaluate(X_test, y_test)[1]*100 , "%")

exit()

import matplotlib.pyplot as plt
epochs = [i for i in range(10)]
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

plt.plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
plt.plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
plt.title('Training & Testing Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

'''
model.add(LSTM(units=128 , return_sequences = True , recurrent_dropout = 0.25 , dropout = 0.25))
model.add(LSTM(units=64 , recurrent_dropout = 0.1 , dropout = 0.1))
model.add(Dense(units = 32 , activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
'''