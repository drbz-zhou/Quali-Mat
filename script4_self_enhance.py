# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 11:25:59 2021

@author: fredr
"""


#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import gc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import scipy.io
import pickle
import toolbox as tools
from data_gen import DataGenerator_mem, DataGenerator_mem_reconstruct
from sklearn.metrics import confusion_matrix
import model_builder
import data_parser as parser
from datetime import datetime

def build_conv_encoder(kernel=5, dropoutrate=0.2):
    model = keras.models.Sequential([
        layers.Conv3D( filters = 10, kernel_size = kernel, padding='same', activation='relu', input_shape=(128, 64 ,50, 1)),
        layers.AveragePooling3D(pool_size=(2, 2, 3)),
        layers.Dropout(dropoutrate),
        layers.Conv3D( filters = 20, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(2, 2, 3)),
        layers.Dropout(dropoutrate),
        layers.Conv3D( filters = 40, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(2, 2, 1)),
        layers.Dropout(dropoutrate),
        layers.Conv3D( filters = 80, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(2, 2, 1)),
        layers.Conv3D( filters = 5, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        #layers.Dropout(dropoutrate),
        #layers.Conv3D( filters = 1, kernel_size = kernel, padding='same', activation='relu'),
        #layers.BatchNormalization(),
    ])
    return model
def build_conv_decoder(kernel=5, dropoutrate=0.2):
    model = keras.models.Sequential([
        #layers.Conv3D( filters = 5, kernel_size = kernel, padding='same', activation='relu'),
        #layers.BatchNormalization(),
        #layers.UpSampling3D((2, 2, 1)),
        layers.Conv3D( filters = 80, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.UpSampling3D((2, 2, 1)),
        layers.Conv3D( filters = 40, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.UpSampling3D((2, 2, 2)),
        layers.Conv3D( filters = 1, kernel_size = kernel, padding='same', activation='relu'),
        #layers.BatchNormalization(),
        #layers.UpSampling3D((2, 2, 5)),
        #layers.Conv3D( filters = 1, kernel_size = kernel, padding='same', activation='relu'),
    ])
    return model


encoder = build_conv_encoder()
decoder = build_conv_decoder()
m_input = keras.Input(shape = (128, 64 ,50, 1)) 
m_output = encoder(m_input)
m_output = decoder(m_output)
model = keras.Model(
    inputs = m_input,
    outputs = m_output,
)

m_ini_learning_rate = 0.0001
m_opt = keras.optimizers.Adam(learning_rate=m_ini_learning_rate)
model.compile(optimizer=m_opt,
              loss=keras.losses.MeanSquaredError(),
              metrics = keras.metrics.MeanSquaredError())
model.summary()
encoder.summary()
decoder.summary()


#%% setup GPU
tools.tf_setup_GPU()
#tools.tf_mem_patch()
numClass = 47
plim = 13 # 7 for 6 people (64GB RAM), 13 for all 12 people (needs 128GB RAM), 2 for 1 person with quick test

params = {'batch_size': 250, 'shuffle': True, 'n_classes': numClass}
params['dim'] = (128, 64, 50)
params['n_channels'] = 1
params['datapath'] = '../Data/SessionCSV/'
params['y_offset']=1  #first label is 1 instead of 0
params['labelpath']='../Data/labels_50_10/' # _50_5 or _50_10
out_path = '../Outputs/'


f=np.load(params['labelpath']+'LabelMeta'+str(numClass)+'.npz')
label_lens=f['arr_0']
Meta_Ind = f['arr_1'] # P, R, Slices, y-47, y-9
if numClass == 9:
    labels=Meta_Ind[:,4]  #3-47, 4-9
elif numClass == 47:
    labels=Meta_Ind[:,3]  #3-47, 4-9

#%% load training data into memory
print('load training data')
mDataDict = {}
for P in range(1,plim): #1,13
    for R in range(1,4): # 1,4
        #if R!= out_Session:
            datastr = 'P'+str(P)+'R'+str(R)
            filename = params['datapath'] + datastr + '.npy'
            print(datastr)
            mDataDict[datastr]=np.load(filename)
# load slice label and frame index data into memory
mSliceDict = {}
for P in range(1,plim):
    for R in range(1,4):
        datastr = 'P'+str(P)+'R'+str(R)
        mSliceDict[datastr]=np.genfromtxt(
            params['labelpath']+datastr+'_label_'+str(numClass)+'b.csv', delimiter=',',dtype=int)
out_Session = 1
train_list = np.where( (Meta_Ind[:,1]!=out_Session) & 
                      (Meta_Ind[:,0]<plim) )[0].tolist()
test_list = np.where( (Meta_Ind[:,1]==out_Session)  & 
                     (Meta_Ind[:,0]<plim) )[0].tolist()

test_subind, valid_subind = tools.train_valid_split_jump(np.array(test_list), 10)  #train_list
train_list_ind = train_list#np.array(train_list)[train_subind].tolist()
valid_list_ind = np.array(test_list)[valid_subind].tolist()  #train_list
test_list_ind = np.array(test_list)[test_subind].tolist()

tools.print_time()        

#%%



acc = []
val_acc = []
loss = []
val_loss = []
epoch = 10
modelsavefile = '../Outputs/model_ConvAutoEncoder.h5'
patience= 100

#%
train_gen = DataGenerator_mem_reconstruct(train_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict,**params)
#valid_gen = DataGenerator_mem_reconstruct(valid_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params)
# train model
history = model.fit( x = train_gen, epochs = epoch, batch_size=params['batch_size'],
              #use_multiprocessing = True,
              #validation_data = m_datagen_valid,
              #callbacks=[cb_checkpoint, cb_earlystop, cb_learningrate, cb_tensorboard], #, cb_learningrate
              verbose = 1 # 2 if log to file
        )

#%% build the classification model
enc_trained = model.layers[1]
enc_trained.trainable = False
enc_trained.summary()
m_input = keras.Input(shape = (128, 64 ,50, 1)) 
m_output = encoder(m_input)
m_output = layers.Dropout(0.2)(m_output)
m_output = layers.Conv3D( filters = 1, kernel_size = 5, padding='same', activation='relu')(m_output)
m_output = layers.BatchNormalization()(m_output)
m_output = layers.Reshape((8,4,5))(m_output)
m_output = layers.Conv2D( filters = 1, kernel_size = 5, padding='same', activation='relu')(m_output)
m_output = layers.Flatten()(m_output)
m_output = layers.Dense(numClass, activation='softmax')(m_output)


model_cls = keras.Model(
    inputs = m_input,
    outputs = m_output,
)

m_ini_learning_rate = 0.0001
m_opt = keras.optimizers.Adam(learning_rate=m_ini_learning_rate)
model_cls.compile(optimizer=m_opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model_cls.summary()

model_arch="Conv3D"
acc = []
val_acc = []
loss = []
val_loss = []
epoch = 1000
modelsavefile = '../Outputs/model_'+model_arch+'_'+str(numClass)+'.h5'
patience= 100

#%
train_generator = DataGenerator_mem(train_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict,**params)
valid_generator = DataGenerator_mem(valid_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params)
# train model
model_cls, history = tools.train_gen(
    model_cls, epoch, train_generator, valid_generator, modelsavefile, patience, Batch_size=params['batch_size'], 
    initial_learning_rate=m_ini_learning_rate, logpath = '../Outputs/Logs/'+model_arch+'_LOS'+str(out_Session)+'_')
acc, val_acc, loss, val_loss = tools.append_history(
    history, acc, val_acc, loss, val_loss)

tools.print_time()

#% test model
model_cls.load_weights(modelsavefile) #model needs to be built first 
params_test = params.copy()
params_test['batch_size'] = 1
params_test['shuffle'] = False

test_gen = DataGenerator_mem(test_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params_test)
m_y_test = labels[test_list_ind]-1

m_y_pred = model_cls.predict(test_gen)
acc_test = sum(m_y_test == np.argmax(m_y_pred, axis=1)) / m_y_test.shape[0]

cm = confusion_matrix(m_y_test, np.argmax(m_y_pred, axis=1))
print(acc_test)
print(cm)
#%
tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path = out_path + model_arch + '_' + \
        str(numClass) + '_LO' + str(out_Session)+'_')
tools.plot_acc_loss(acc, val_acc, loss, val_loss, file_path = out_path + model_arch + '_' + \
        str(numClass) + '_LO' + str(out_Session)+'_')