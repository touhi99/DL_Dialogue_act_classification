import numpy as np
import pandas as pd
import _pickle as cPickle
from collections import defaultdict
import re
import sys
import os
from functools import reduce
import gzip

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.models import Sequential
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.callbacks import History
from keras.models import load_model

import matplotlib.pyplot as plt


MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.1
directory = '/mount/arbeitsdaten31/studenten1/deeplearning/2017/infy/text/'

data_train = pd.read_csv(directory+"train.txt", sep='\t',header=None)
print(data_train.shape)

data_dev = pd.read_csv(directory+"dev.txt", sep='\t',header=None)
print(data_dev.shape)

text = data_train[data_train.columns[2]].tolist()
print(len(text))

text_dev = data_dev[data_dev.columns[2]].tolist()
print(len(text_dev))

labels = data_train[data_train.columns[1]].tolist()
print(len(labels))

labels_dev = data_dev[data_dev.columns[1]].tolist()
print(len(labels_dev))

tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

word_index['UNK'] = 9754

data_test = pd.read_csv(directory+"test.txt", sep='\t',header=None)
print(data_test.shape)

text_test = data_test[data_test.columns[2]].tolist()
print(len(text_test))

labels_test = data_test[data_test.columns[1]].tolist()
print(len(labels_test))

sequences_test = []           
for sentence in text_test:
    words = sentence.lower().split()
    sent = []
    for w in words:
        #print w
        if w not in word_index.keys():
            sent.append(word_index['UNK'])
        else:
            sent.append(word_index[w])
    #print sent
    sequences_test.append(sent)   
    if len(words) > MAX_SEQUENCE_LENGTH:
        MAX_SEQUENCE_LENGTH = len(words)

sequences_dev = []
print("Len: "+str(len(text_dev)))        
for sentence in text_dev:
    words = sentence.lower().split()
    sent = []
    for w in words:
        if w not in word_index.keys():
            sent.append(word_index['UNK'])
        else:
            sent.append(word_index[w])
    sequences_dev.append(sent)   
    if len(words) > MAX_SEQUENCE_LENGTH:
        MAX_SEQUENCE_LENGTH = len(words)
print(len(sent))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
data_dev = pad_sequences(sequences_dev, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
labels_test = to_categorical(np.asarray(labels_test))
labels_dev = to_categorical(np.asarray(labels_dev))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print('Shape of data tensor:', data_test.shape)
print('Shape of label tensor:', labels_test.shape)
print('Shape of data tensor:', data_dev.shape)
print('Shape of label tensor:', labels_dev.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data
print(x_train)
print(x_train.shape)

y_train = labels
print(y_train[0])

print('Traing and validation set number of data')
print(y_train.sum(axis=0))
#print(y_val.sum(axis=0))

GLOVE_DIR = "GloveEmbeddings"

embeddings_index = {}
f = open(os.path.join(directory+'glove.twitter.27B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

model = Sequential()
model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
model.add(Dropout(0.25))
model.add(LSTM(100))

model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])
print("model fitting - LSTM")
model.summary()

history = model.fit(x_train, y_train, validation_data=(data_dev, labels_dev), nb_epoch=5)

# batch_size=50
model.save("LSTM-Model-mini-batch_f") 

#model = load_model("LSTM-Model-2")

print(history.history.keys())  

plt.figure(1)  
   
 # summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'val'], loc='upper left')  
   
 # summarize history for loss  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'val'], loc='upper left')  
plt.show() 

print('Shape of data tensor:', data_test.shape)
print('Shape of label tensor:', labels_test.shape)

print('Test number of data')
print (labels_test.sum(axis=0))

scores = model.evaluate(data_test, labels_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

