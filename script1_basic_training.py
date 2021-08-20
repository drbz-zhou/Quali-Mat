# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:20:23 2021

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
from data_gen import DataGenerator_mem
from sklearn.metrics import confusion_matrix
import model_builder
import data_parser as parser
 
# setup GPU
tools.tf_setup_GPU()
#tools.tf_mem_patch()
numClass = 47
bathsize = 200
out_Session = 3
plim = 7 # 7 for 6 people (64GB RAM), 13 for all 12 people (needs 128GB RAM)

out_path = '../Outputs/'
params = {'batch_size': bathsize, 'shuffle': True, 'n_classes': numClass}
params['dim'] = (128, 64, 50)
params['n_channels'] = 1
params['datapath'] = '../Data/SessionCSV/'
params['labelpath']='../Data/labels_50_10/'
f=np.load(params['labelpath']+'LabelMeta'+str(numClass)+'.npz')
label_lens=f['arr_0']
Meta_Ind = f['arr_1'] # P, R, Slices, y-47, y-9
if numClass == 9:
    labels=Meta_Ind[:,4]  #3-47, 4-9
elif numClass == 47:
    labels=Meta_Ind[:,3]  #3-47, 4-9
    
#train_list = np.arange(12807).tolist()
#test_list = np.arange(12808,18216).tolist()#

# leave session out
train_list = np.where( (Meta_Ind[:,1]!=out_Session) & (Meta_Ind[:,0]<plim) )[0].tolist()
test_list = np.where( (Meta_Ind[:,1]==out_Session)  & (Meta_Ind[:,0]<plim) )[0].tolist()

test_subind, valid_subind = tools.train_valid_split_jump(np.array(test_list), 10)  #train_list
train_list_ind = train_list#np.array(train_list)[train_subind].tolist()
valid_list_ind = np.array(test_list)[valid_subind].tolist()  #train_list
test_list_ind = np.array(test_list)[test_subind].tolist()
# load training data into memory
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
for P in range(1,13):
    for R in range(1,4):
        datastr = 'P'+str(P)+'R'+str(R)
        mSliceDict[datastr]=np.genfromtxt(
            params['labelpath']+datastr+'_label_'+str(numClass)+'b.csv', delimiter=',',dtype=int)
#%%            
train_gen = DataGenerator_mem(train_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict,**params)
valid_gen = DataGenerator_mem(valid_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params)

tools.print_time()
#for i in range(10):
#    mDataExample=train_gen.__getitem__(i)
#    tools.print_time()
#%% check if the list is ok, especially if leave P or R out is properly assigned
# PRS = np.zeros((3,len(train_subind)))
# for i in range(len(train_subind)):
#     P, R, S = parser.decode_index_mem(train_subind[i], Meta_Ind)
#     PRS[:,i] = [P,R,S]
#%%
#mDataExample=train_gen.__data_generation([10])
#model = model_builder.build_Conv3D(filters=5, kernel=3, dense=256, numClass=numClass)
#model = model_builder.build_TConv_Incpt(filters = 5, kernel = (1,1,3), fine_tune_at = 500, numClass = numClass)
model_arch="Conv3D"
if model_arch=="Conv3D":
    model = model_builder.build_Conv3D(
        filters=5, kernel=5, dense=512, numClass=numClass, dropoutrate = 0.5)
elif model_arch=="TConv_Imgnet":
    model = model_builder.build_TConv_Imgnet(
        filters = 5, kernel = (1,1,3), fine_tune_at = 200, numClass = numClass, imag_model = 'EfficientNetB0')
elif model_arch=="Img_Tconv":
    model = model_builder.build_Img_TConv(numClass=numClass)
elif model_arch=="Img_Tconv_TD":
    model = model_builder.build_Img_TConv_TD(numClass=numClass)
    
m_opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=m_opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

acc = []
val_acc = []
loss = []
val_loss = []
epoch = 10000
modelsavefile = 'Outputs/model_'+model_arch+'_'+str(numClass)+'.h5'
patience= 50

# train model
model, history = tools.train_gen(
    model, epoch, train_gen, valid_gen, modelsavefile, patience, Batch_size=bathsize)
acc, val_acc, loss, val_loss = tools.append_history(
    history, acc, val_acc, loss, val_loss)

tools.print_time()
#%% test model
model.load_weights(modelsavefile) #model needs to be built first 
params['batch_size'] = 1
params['shuffle'] = False
# load testing data into memory
print('load testing data')
del train_gen
del valid_gen
del mDataDict
gc.collect()
mDataDict = {}
for P in range(1,plim): #1,13
    R = out_Session
    datastr = 'P'+str(P)+'R'+str(R)
    filename = params['datapath'] + datastr + '.npy'
    print(datastr)
    mDataDict[datastr]=np.load(filename)
            
test_gen = DataGenerator_mem(test_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params)
m_y_test = labels[test_list_ind]-1

m_y_pred = model.predict(test_gen)
acc_test = sum(m_y_test == np.argmax(m_y_pred, axis=1)) / m_y_test.shape[0]

cm = confusion_matrix(m_y_test, np.argmax(m_y_pred, axis=1))
print(acc_test)
print(cm)
#%%
tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path = out_path + model_arch + '_' + \
        str(numClass) + '_LO' + str(out_Session)+'_')
tools.plot_acc_loss(acc, val_acc, loss, val_loss, file_path = out_path + model_arch + '_' + \
        str(numClass) + '_LO' + str(out_Session)+'_')
    
