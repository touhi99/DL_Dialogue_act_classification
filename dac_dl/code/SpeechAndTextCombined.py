from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import os

import keras
import tensorflow as tf
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Average
from keras.layers import Embedding, merge, Dropout, LSTM, GRU, Bidirectional
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Model
from keras.models import load_model
from keras.models import Sequential
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras.callbacks import History
from keras.utils import plot_model
from keras import optimizers
import gzip
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

from SpeechCNNModel import CNNModelKeras as CNN

#pros ensemble concat 0.8163736977924904
#MFCC ensemble concat 0.811653645709157

#pros ensemble avg 0.770377604290843
#mfcc ensemble avg 0.706673176959157

#pros ensemble max 0.7433593751241764
#mfcc ensemble max 0.768684895709157

#pros ensemble add 0.8158203127483526
#mfcc ensemble add 0.7914713540424904

class Combined():
	def __init__(self, feature_name, window_size):
		K.set_image_dim_ordering('th')
		self.train_set_batch = 445
		self.test_set_batch = 97
		self.maxlen =  1000 #pros 3361, mfcc 3360
		self.window_size = window_size #13 for MFCC, 15 for Prosodic
		self.batch_size = 64
		self.path = '/mount/arbeitsdaten31/studenten1/deeplearning/2017/infy/'
		self.directory = '/mount/arbeitsdaten31/studenten1/deeplearning/2017/infy/text/'
		self.feature_name = feature_name
		#self.ensembleModel()
		self.entrain()

	#our main ensemble, train rnn and cnn and merge them
	def entrain(self):
		
		MAX_SEQUENCE_LENGTH = 100
		MAX_NB_WORDS = 20000
		EMBEDDING_DIM = 50
		VALIDATION_SPLIT = 0.1

		data_train = pd.read_csv(self.directory+"train_up.txt", sep='\t',header=None)
		print(data_train.shape)

		data_dev = pd.read_csv(self.directory+"dev_up.txt", sep='\t',header=None)
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
		data_test = pd.read_csv(self.directory+"test_up.txt", sep='\t',header=None)
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
		for sentence in text_dev:
		    words = sentence.lower().split()
		    sent = []
		    for w in words:
		        #print w
		        if w not in word_index.keys():
		            sent.append(word_index['UNK'])
		        else:
		            sent.append(word_index[w])
		    #print sent
		    sequences_dev.append(sent)   
		    if len(words) > MAX_SEQUENCE_LENGTH:
		        MAX_SEQUENCE_LENGTH = len(words)

		data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
		data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
		data_dev = pad_sequences(sequences_dev, maxlen=MAX_SEQUENCE_LENGTH)
		print('Shape of data tensor:', data_dev.shape)

		labels = to_categorical(np.asarray(labels))
		labels_test = to_categorical(np.asarray(labels_test))
		labels_dev = to_categorical(np.asarray(labels_dev))

		indices = np.arange(data.shape[0])

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
		f = open(os.path.join(self.directory+'glove.twitter.27B.50d.txt'))
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
		#RNN Model
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

		#CNN Model
		model2 = Sequential()
		model2.add(Convolution2D(32, (5, 5), activation='relu', input_shape=(1,self.maxlen,self.window_size)))
		model2.add(MaxPooling2D(pool_size=(5,1)))
		model2.add(Dropout(0.25))

		model2.add(Convolution2D(32, (5, 2))) #(5,2) for MFCC, (5,3) for Prosodic
		model2.add(MaxPooling2D(pool_size=(4,2)))
		model2.add(Dropout(0.25))

		model2.add(Convolution2D(32, (5, 2))) #(5,2) for MFCC, (5,3) for Prosodic
		model2.add(MaxPooling2D(pool_size=(4,2)))
		model2.add(Dropout(0.25))

		model2.add(Flatten())
		model2.add(Dense(100, activation='relu'))
		model2.add(Dropout(0.5))
		model2.add(Dense(4, activation='softmax'))

		#adam = optimizers.Adam(lr=0.003)
		model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model2.summary()

		model1_input = Input(shape=(100,))
		model2_input = Input(shape=(1,self.maxlen,self.window_size))
		
		#Get individual models output
		model1_output = model(model1_input)
		model2_output = model2(model2_input)

		input_path = self.path+'train_set/'+self.feature_name+'/Input/'
		output_path = self.path+'train_set/'+self.feature_name+'/Output/'

		dev_input = self.path+'dev_set/'+self.feature_name+'/Input/'
		dev_output = self.path+'dev_set/'+self.feature_name+'/Output/'

		#separate functions Concat/Avg/Add/Max
		merged = keras.layers.concatenate([model1_output, model2_output])
		#merged = keras.layers.average([model1_output, model2_output])
		#merged = keras.layers.add([model1_output, model2_output])
		#merged = keras.layers.maximum([model1_output, model2_output])
		merged = Dense(4, activation='softmax')(merged)

		combined_model = Model(inputs=[ model1_input, model2_input ], output=merged)
		combined_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		combined_model.summary()

		acc_set = list(range(5))
		loss_set = list(range(5))
		
		for j in range(5):
			acc = 0
			loss = 0
			for i in range(1, self.train_set_batch):
				f = gzip.GzipFile(input_path+'pros_in_'+str(i)+'.npy.gz',"r")
				X_train = np.load(f)
				f.close()
				''' for reduced dim'''
				X_train = X_train[:,:1000,]

				f = gzip.GzipFile(output_path+'pros_out_'+str(i)+'.npy.gz',"r")
				Y_train = np.load(f)
				f.close()

				#print(Y_train)
				X_train = X_train.reshape(X_train.shape[0], 1, self.maxlen, self.window_size)

				f = gzip.GzipFile(dev_input+'pros_in_'+str(i)+'.npy.gz',"r")
				X_dev = np.load(f)
				f.close()
				''' for reduced dim'''
				X_dev = X_dev[:,:1000,]

				f = gzip.GzipFile(dev_output+'pros_out_'+str(i)+'.npy.gz',"r")
				Y_dev = np.load(f)
				f.close()
				X_dev = X_dev.reshape(X_dev.shape[0], 1, self.maxlen, self.window_size)
				
				print(str(j+1)+"-"+str(i))
				#train on combined model using mix data
				if(i==self.train_set_batch-1):
					history = combined_model.fit([ x_train[(i-1)*self.batch_size : (i*self.batch_size)-31], X_train[0:32] ], Y_train[0:32], validation_data=( [ data_dev[(i-1)*13 : i*13]  , X_dev] , Y_dev ), epochs=1)
				else:
					history = combined_model.fit([ x_train[(i-1)*self.batch_size : i*self.batch_size], X_train], Y_train, validation_data=( [ data_dev[(i-1)*13 : i*13]  , X_dev] , Y_dev ), epochs=1)
				acc+= history.history['acc'][0]
				loss+=history.history['loss'][0]
				
			acc_set[j]=acc/(self.train_set_batch-1)
			loss_set[j]=loss/(self.train_set_batch-1)
		model.save("ensemble-Model-concat-mfcc")
		#Test on combined model
		self.ensembleModel(combined_model)

		matplotlib.use('Agg')
		plt.figure(1)  
 		# summarize history for accuracy
		plt.subplot(211)  
		plt.plot(acc_set)  
		plt.title('model accuracy')  
		plt.ylabel('accuracy')  
		plt.xlabel('epoch')  
		plt.legend(['train', 'test'], loc='upper left')  
		   
		 # summarize history for loss  
		
		plt.subplot(212)  
		plt.plot(loss_set)  
		plt.title('model loss')  
		plt.ylabel('loss')  
		plt.xlabel('epoch')  
		plt.legend(['train', 'test'], loc='upper left')  
		plt.show()

	def ensembleModel(self, combined_model):
		#self.textModel = load_model("LSTM-Model")
		#self.MFCCModel = load_model("MFCC-Model")
		#self.MFCCModel = load_model("Prosodic-Model")
		#self.MFCCModel = load_model("MFCC-Model-reduced")
		data_test, labels_test = self.textData()
		#test on combined model
		accuracy = 0
		loss = 0
		for i in range(1,self.test_set_batch):
			X_test, Y_test = self.speechData(i)
			text_data = data_test[(i-1)*self.batch_size:(i*self.batch_size)]
			text_label = labels_test[(i-1)*self.batch_size:(i*self.batch_size)]

			if(i==self.test_set_batch-1):
				scores = combined_model.evaluate([text_data[0:10], X_test[0:10]], Y_test[0:10], verbose=1)
			else:
				scores = combined_model.evaluate([text_data, X_test], Y_test, verbose=1)

			loss += scores[0]
			accuracy += scores[1]

		print("Accuray: "+str(accuracy/(self.test_set_batch-1)))

	#deprecated, previous only test on combined model, not trained
	def ensemble(self, output, text_output, speech_output):
		#Prosodic
		#average acc .8121
		#weighted avg acc .8174
		#maxpool acc .8081

		#MFCC
		#average acc .8119
		#weighted avg acc .8195
		#maxpool acc .8157

		#Text
		#Acc .8210
		average = np.mean([text_output, speech_output], axis=0)
		w_avg = np.average([text_output, speech_output], axis=0, weights=[0.6,0.4])
		maxpool = np.amax([text_output, speech_output], axis=0)
		acc = (np.mean(np.equal(np.argmax(w_avg, 1), np.argmax(output, 1))))
		print(acc)
		return acc
	#deprecated
	def textData(self):
		MAX_SEQUENCE_LENGTH = 100

		data_train = pd.read_csv("train_up.txt", sep='\t',header=None)
		text = data_train[data_train.columns[2]].tolist()
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(text)
		word_index = tokenizer.word_index

		data_test = pd.read_csv("test_up.txt", sep='\t',header=None)
		print(data_test.shape)
		text_test = data_test[data_test.columns[2]].tolist()
		print(len(text_test))
		labels_test = data_test[data_test.columns[1]].tolist()
		print(len(labels_test))
		
		word_index['UNK'] = 9754
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

		data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
		labels_test = to_categorical(np.asarray(labels_test))
		return (data_test, labels_test)
	#deprecated
	def speechData(self, i):
		f = gzip.GzipFile(self.path+'test_set/'+self.feature_name+'/Input/'+'pros_in_'+str(i)+'.npy.gz',"r")
		X_test = np.load(f)
		f.close()
		f = gzip.GzipFile(self.path+'test_set/'+self.feature_name+'/Output/'+'pros_out_'+str(i)+'.npy.gz',"r")
		Y_test = np.load(f)
		f.close()
		'''reduced dim'''
		X_test = X_test[:,:1000,]
		X_test = X_test.reshape(X_test.shape[0], 1, self.maxlen, self.window_size)
		return (X_test, Y_test)

if __name__ == "__main__":
	Combined('MFCC',13)
	#Combined('Prosodic',15)