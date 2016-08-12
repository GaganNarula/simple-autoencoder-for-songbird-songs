execfile("core_functions.py")
execfile("mfcc.py")

import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Flatten, Dropout, Activation

import time

freq_dim = 120
time_dim = 15
n_dim = freq_dim*time_dim
nb_classes = 2
nb_epoch = 15

x2 = np.asarray(pd.read_csv("data/b12r14-2010-11-28-data_cnn_gap_winlen15.csv"))
y2 = np.asarray(pd.read_csv("data/b12r14-2010-11-28-label_cnn_gap_winlen15.csv"))
y2 = y2[:,(time_dim-1)/2]
y2 = np.expand_dims(y2,axis=1)
xy = np.hstack((x2,y2))
np.random.shuffle(xy)

x2 = xy[:,0:n_dim]
y2 = xy[:,n_dim]

x1 = data_rshp_exp(x2,nb_time = time_dim, nb_spec = freq_dim)
y1 = np.sign(y2)
y1 = to_utils(nb_classes,y1)
x1 = np.expand_dims(x1,axis=1)

x_train, y_train, x_test, y_test = data_split(x1,y1,0.8,0.2)[0:4]



input = Input(shape=(1,time_dim,freq_dim))
conv1 = Convolution2D(32,3,3,activation='relu')
dropout1 = Dropout(0.3)
conv2 = Convolution2D(32,3,3,activation='relu')
dropout2 = Dropout(0.3)
pool1 = MaxPooling2D()
conv3 = Convolution2D(64,3,3,activation='relu')
dropout3 = Dropout(0.3)
pool2 = MaxPooling2D((1,2))
flat = Flatten()
dense1 = Dense(100,activation='relu')
dropout4 = Dropout(0.3)
dense2 = Dense(nb_classes,activation='softmax')

model = Model(input=input,output=dense2(dropout4(dense1(flat(pool2(dropout3(conv3(pool1(dropout2(conv2(dropout1(conv1(input)))))))))))))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,nb_epoch=nb_epoch)
score = model.evaluate(x_test,y_test)
print('Loss:',score[0])
print('Accuracy:',score[1])

json_string = model.to_json()
open('cgd_v2_architecture.json','w').write(json_string)
model.save_weights('cgd_v2_weights.h5')


x2 = np.asarray(pd.read_csv("data/g5r15-2011-02-16-data_cnn_gap_winlen15.csv"))
y2 = np.asarray(pd.read_csv("data/g5r15-2011-02-16-label_cnn_gap_winlen15.csv"))
y2 = y2[:,(time_dim-1)/2]
y2 = np.expand_dims(y2,axis=1)
xy = np.hstack((x2,y2))
np.random.shuffle(xy)

x2 = xy[:,0:n_dim]
y2 = xy[:,n_dim]

x1 = data_rshp_exp(x2,nb_time = time_dim, nb_spec = freq_dim)
y1 = np.sign(y2)
y1 = to_utils(nb_classes,y1)
x1 = np.expand_dims(x1,axis=1)

score1 = model.evaluate(x1,y1)
print('Loss on g5r15:',score1[0])
print('Accuracy on g5r15:',score1[1])
