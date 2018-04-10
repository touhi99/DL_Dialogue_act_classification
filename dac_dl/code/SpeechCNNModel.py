import numpy as np
from collections import OrderedDict
import gzip
import tensorflow as tf
from random import randint
import random
import itertools, operator, random
from collections import defaultdict
from functools import reduce
import matplotlib.pyplot as plt

from keras import backend as K
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import History
from keras import optimizers
import matplotlib
import matplotlib.pyplot as plt

class Preprocessing():
	def __init__(self):
		self.uttr_no = {}
		self.labels=[]
		self.batch = 64 #dev 13, rest 64
		self.CW_size = 1 #2 for pros
		self.max_length=3360
		self.f_content=13
		self.no_of_label = 4
		self.path = '/mount/arbeitsdaten31/studenten1/deeplearning/2017/infy/train_set/'
		#self.shuffle('MFCC','_dev')
		self.zero_padding('MFCC','')
		self.context_window('MFCC','')
		#maxl - #total - f - f_type
		#3361 - 28384 - pros train
		#3360 - 28384 - MFCC train

	#shuffle the data to align with speech and text
	def shuffle(self, feature, f_type):
		lines = defaultdict(list)
		lines2 = defaultdict(list)
		with open(self.path+'MFCC'+'_Features'+f_type+'.arff', "r") as in_file:
			for line in in_file:
				s_line = line.split(",")
				lines[s_line[0]].append(line)

		with open(self.path+'Prosodic'+'_Features'+f_type+'.arff', "r") as in_file:
			for line in in_file:
				s_line = line.split(",")
				lines2[s_line[0]].append(line)

		# Randomize the order
		rnd_keys = random.sample(lines.keys(), len(lines))

		print(len(rnd_keys))
		with open(self.path+'_random'+'_utter_list'+f_type+'.arff', "w") as out_file:
			for k in rnd_keys:
				out_file.write(k+"\n")

		# Write back to the file?
		with open(self.path+'MFCC'+'_Features_2'+f_type+'.arff', "w") as out_file:
			for k in rnd_keys:
				for line in lines[k]:
					out_file.write("{}".format(line))

		with open(self.path+'Prosodic'+'_Features_2'+f_type+'.arff', "w") as out_file:
			for k in rnd_keys:
				for line in lines2[k]:
					out_file.write("{}".format(line))

	#pad the data to its max length
	def zero_padding(self, feature, f_type):
		feature_reader = open(self.path+feature+'_Features'+'_2'+f_type+'.arff','r')
		feature_reader = feature_reader.read().split('\n')

		for line in feature_reader:
			if not line.startswith('@') and len(line) > 0:
				line = line.split(',')
				if line[0] not in self.uttr_no:
					self.uttr_no[line[0]] = 1
				else:
					self.uttr_no[line[0]] += 1
		
		key_max = max(self.uttr_no.keys(), key=(lambda k: self.uttr_no[k]))

		n = np.zeros((len(self.uttr_no), self.max_length, self.f_content))
		print(self.max_length) 
		print(len(self.uttr_no))
		
		u_count = 0
		f_count = 0
		label = 0
		for line in feature_reader:
			if not line.startswith('@') and len(line) > 0:
				line = line.split(',')
				f_length = self.uttr_no[line[0]]
				label = int(line[-1])

				self.labels.append((line[0],label))
				for i in range(0,self.f_content):
					n[u_count,f_count,i] = line[i+1]
				f_count += 1
				if f_count==f_length:
					f_count=0
					u_count += 1
		self.labels = self.orderedSet(self.labels)
		
		f = gzip.GzipFile(self.path+'zero_padded_mfcc'+f_type+'.npy.gz',"w")
		np.save(f, n)
		f.close()

	#add context window for Prosodic, for MFCC it's just 1 window per sample
	def context_window(self, feature, f_type):
		f = gzip.GzipFile(self.path+'zero_padded_mfcc'+f_type+'.npy.gz',"r")
		data = np.load(f)
		f.close()

		final = np.zeros( (self.batch,self.max_length,self.f_content) )
		output = np.zeros( (self.batch, self.no_of_label), int )
		k=0
		name=1

		for i in range(0, len(self.uttr_no)):
			for j in range(0, self.max_length):
				
				#for Prosodic with CW, below part
				'''
				for x in range(-self.CW_size, self.CW_size+1):
					if(j+x<0 or j+x>=self.max_length):
						if x==-self.CW_size:
							cw_save = np.zeros(self.f_content)
						else:
							cw_save = np.concatenate( (cw_save, np.zeros(self.f_content)) )
					else:
						if x==-self.CW_size:
							cw_save = np.array(data[i][j+x])
						else:
							cw_save = np.concatenate( (cw_save, np.array(data[i][j+x]) ))
				'''
				final[k][j] = data[i][j]
				#final[k][j] = cw_save
				cw_save = None
			output[k][self.labels[i][1]] = 1
			k+=1

			if(k%self.batch==0):
				print('batch '+str(name)+' saved')
				f = gzip.GzipFile(self.path+feature+'/Input/pros_in_'+str(name)+'.npy.gz',"w")
				np.save(f, final)
				f.close()
				final = np.zeros( (self.batch,self.max_length,self.f_content) )

				f = gzip.GzipFile(self.path+feature+'/Output/pros_out_'+str(name)+'.npy.gz',"w")
				np.save(f, output)
				f.close()
				output = np.zeros( (self.batch, self.no_of_label), int )

				k=0
				name+=1
		f = gzip.GzipFile(self.path+feature+'/Input/pros_in_'+str(name)+'.npy.gz',"w")
		np.save(f, final)
		f.close()
		
		f = gzip.GzipFile(self.path+feature+'/Output/pros_out_'+str(name)+'.npy.gz',"w")
		np.save(f, output)
		f.close()

	def orderedSet(self, seq):
		seen = set()
		seen_add = seen.add
		return [x for x in seq if not (x in seen or seen_add(x))]

#Primary CNN model class
class CNNModelKeras():
	def __init__(self, feature_name, max_l, window_size, batch_tuple):
		K.set_image_dim_ordering('th')
		self.maxlen = max_l # 1000 for reduced dim., 3360 for mfcc, 3361 for Prosodic
		self.batch_size = 64
		self.nb_classes = 4
		self.feature_name = feature_name
		self.window_size = window_size
		self.train_set_batch = batch_tuple[0]
		self.test_set_batch = batch_tuple[1]
		self.path = '/mount/arbeitsdaten31/studenten1/deeplearning/2017/infy/'
		self.train_summary = {}
		self.test_summary = {}
		self.model()

	def model(self):
		input_path = self.path+'train_set/'+self.feature_name+'/Input/'
		output_path = self.path+'train_set/'+self.feature_name+'/Output/'

		dev_path = self.path+'dev_set/'+self.feature_name

		model = Sequential()
		#commented out actual dimension network settings
		''' actual dim'''
		'''
		model.add(Convolution2D(32, 5, 5, activation='relu', input_shape=(1,self.maxlen,self.window_size)))
		model.add(MaxPooling2D(pool_size=(10,1)))
		model.add(Dropout(0.25))

		model.add(Convolution2D(32, 5, 2)) #(5,2) for MFCC, (5,3) for Prosodic
		model.add(MaxPooling2D(pool_size=(4,2)))
		model.add(Dropout(0.25))
		model.add(Convolution2D(32, 5, 2)) #(5,2) for MFCC, (5,3) for Prosodic
		model.add(MaxPooling2D(pool_size=(4,2)))
		model.add(Dropout(0.25))

		model.add(MaxPooling2D(pool_size=(5,1)))
		model.add(Dropout(0.25))
		'''

		''' reduced dim '''
		
		model.add(Convolution2D(32, (5, 5), activation='relu', input_shape=(1,self.maxlen,self.window_size)))
		model.add(MaxPooling2D(pool_size=(5,1)))
		model.add(Dropout(0.25))

		model.add(Convolution2D(32, (5, 2))) #(5,2) for MFCC, (5,3) for Prosodic
		model.add(MaxPooling2D(pool_size=(4,2)))
		model.add(Dropout(0.25))

		model.add(Convolution2D(32, (5, 2))) #(5,2) for MFCC, (5,3) for Prosodic
		model.add(MaxPooling2D(pool_size=(4,2)))
		model.add(Dropout(0.25))
		
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(4, activation='softmax'))

		#we explored variaous value for learning rate
		#adam = optimizers.Adam(lr=0.003)
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.summary()

		#shuffled_batch = random.sample(range(1, self.train_set_batch), self.train_set_batch-1)
		acc_set = list(range(5))
		loss_set = list(range(5))
		#tried out 5/10 epochs
		for j in range(5):
			acc = 0
			loss = 0
		
			for i in range(1,self.train_set_batch):
				f = gzip.GzipFile(input_path+'pros_in_'+str(i)+'.npy.gz',"r")
				X_train = np.load(f)
				f.close()
				 #for reduced dim
				X_train = X_train[:,:1000,]

				f = gzip.GzipFile(output_path+'pros_out_'+str(i)+'.npy.gz',"r")
				Y_train = np.load(f)
				f.close()
				X_train = X_train.reshape(X_train.shape[0], 1, self.maxlen, self.window_size)

				f = gzip.GzipFile(dev_path+'/Input/'+'pros_in_'+str(i)+'.npy.gz',"r")
				X_val = np.load(f)
				f.close()
				 #for reduced dim
				X_val = X_val[:,:1000,]

				f = gzip.GzipFile(dev_path+'/Output/'+'pros_out_'+str(i)+'.npy.gz',"r")
				Y_val = np.load(f)
				f.close()
				X_val = X_val.reshape(X_val.shape[0], 1, self.maxlen, self.window_size)
				#print(K.get_value(model.optimizer.lr))
				#get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[14].output])
				#layer_output = get_3rd_layer_output([X_train, 1])[0]
				#print(layer_output.shape)
				if(i==self.train_set_batch-1):
					history = model.fit(X_train[0:10], Y_train[0:10], batch_size=64, validation_data=(X_val, Y_val), epochs=1)
				else:
					history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=64, epochs=1)
				#print(history.history['acc'])
				print("Epoch: "+str(j+1)+ " Batch: "+str(i))
				acc+= history.history['acc'][0]
				loss+=history.history['loss'][0]

			acc_set[j]=acc/(self.train_set_batch-1)
			loss_set[j]=loss/(self.train_set_batch-1)
		model.save("Prosodic-Model") 
		#model.save("MFCC-Model") 
		
		total = 0
		accuracy = 0
		
		#evaluate by batch
		for i in range(1,self.test_set_batch):
			f = gzip.GzipFile(self.path+'test_set/'+self.feature_name+'/Input/'+'pros_in_'+str(i)+'.npy.gz',"r")
			X_test = np.load(f)
			f.close()
			f = gzip.GzipFile(self.path+'test_set/'+self.feature_name+'/Output/'+'pros_out_'+str(i)+'.npy.gz',"r")
			Y_test = np.load(f)
			f.close()

			#for reduced dim
			X_test = X_test[:,:1000,]
			X_test = X_test.reshape(X_test.shape[0], 1, self.maxlen, self.window_size)

			#print(layer_output)
			score = model.evaluate(X_test, Y_test, batch_size=64, verbose=1)
			total += score[0]
			accuracy += score[1]*100

		print("Loss: "+str(total/(self.test_set_batch-1)))			
		print("Accuray: "+str(accuracy/(self.test_set_batch-1)))
		
		matplotlib.use('Agg')
		plt.figure(1)  
 		# summarize history for accuracy
		plt.subplot(211)  
		plt.plot(acc_set)  
		plt.title('model accuracy')  
		plt.ylabel('accuracy')  
		plt.xlabel('epoch')  
		plt.legend(['train', 'dev'], loc='upper left')  
		   
		 # summarize history for loss  
		
		plt.subplot(212)  
		plt.plot(loss_set)  
		plt.title('model loss')  
		plt.ylabel('loss')  
		plt.xlabel('epoch')  
		plt.legend(['train', 'dev'], loc='upper left')  
		plt.show()
		
		
if __name__ == "__main__":

	#Prosodic
	#Loss: 0.8198023093864322
	#Accuray: 69.873046875

	#Prosodic Reduced
	#Loss: 0.792986325143526
	#Accuray: 71.01236979166667

	#MFCC 
	#Loss: 0.7660310800808171
	#Accuray: 69.7265625

	#MFCC Reduced
	#Loss: 0.7290516753370563
	#Accuray: 72.03776041666667

	#Preprocessing()
	actual = (445, 97)
	test = (20, 5)
	#CNNModelKeras('MFCC',1000,13, actual)
	CNNModelKeras('Prosodic',1000,15,actual)
