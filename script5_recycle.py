# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 19:10:21 2021

@author: fredr
"""


#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import gc
import tensorflow as tf
import tensorflow_addons as tfa
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
def build_conv_classifier(numClass=47, kernel=5, dropoutrate=0.2):
    model = keras.models.Sequential([        
        layers.Dropout(dropoutrate),
        
        layers.Conv3D( filters = 1, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Reshape((8,4,5)),
        layers.Conv2D( filters = 2, kernel_size = kernel, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(numClass, activation='softmax')
        ])
    return model

def build_dense_classifier(numClass=47, kernel=5, dropoutrate=0.2):
    model = keras.models.Sequential([        
        layers.Dropout(dropoutrate),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(dropoutrate),
        layers.BatchNormalization(),
        layers.Dense(numClass, activation='softmax')
        ])
    return model
#%% setup GPU
tools.tf_setup_GPU()
cm_all=[]
acc_all=[]
out_Session = 2
epoch = 400
patience= 20
plim = 13 # 7 for 6 people (64GB RAM), 13 for all 12 people (needs 128GB RAM), 2 for 1 person with quick test

numClass = 9 #9, 47
params = {'batch_size': 250, 'shuffle': True, 'n_classes': numClass}
params['dim'] = (128, 64, 50)
params['n_channels'] = 1
params['datapath'] = '../Data/SessionCSV/'
params['y_offset']=1  #first label is 1 instead of 0
params['labelpath']='../Data/labels_50_10/' # _50_5 or _50_10
out_path = '../Outputs/'
model_list = []
cm_list = []
acc_list = []
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

numClass = 9
f=np.load(params['labelpath']+'LabelMeta'+str(numClass)+'.npz')
label_lens=f['arr_0']
Meta_Ind = f['arr_1'] # P, R, Slices, y-47, y-9
labels=Meta_Ind[:,4]  #3-47, 4-9

# load slice label and frame index data into memory
mSliceDict = {}
for P in range(1,plim):
    for R in range(1,4):
        datastr = 'P'+str(P)+'R'+str(R)
        mSliceDict[datastr]=np.genfromtxt(
            params['labelpath']+datastr+'_label_'+str(numClass)+'b.csv', delimiter=',',dtype=int)

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

enc_layers = build_conv_encoder()
cls_layers_9 = build_conv_classifier(numClass, kernel=5, dropoutrate=0.2)
m_input = keras.Input(shape = (128, 64 ,50, 1)) 
m_output = enc_layers(m_input)
m_output = cls_layers_9(m_output)

model_cls_9 = keras.Model(
    inputs = m_input,
    outputs = m_output,
)

m_ini_learning_rate = 0.0001
m_opt = keras.optimizers.Adam(learning_rate=m_ini_learning_rate)

model_cls_9.compile(optimizer=m_opt,
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model_cls_9.summary()

modelsavefile = '../Outputs/model_Conv3D'+'_'+str(numClass)+'.h5'
#%
train_generator = DataGenerator_mem(train_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict,**params)
valid_generator = DataGenerator_mem(valid_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params)
model_cls_9, history = tools.train_gen(
    model_cls_9, epoch, train_generator, valid_generator, modelsavefile, patience, Batch_size=params['batch_size'], 
    initial_learning_rate=m_ini_learning_rate, logpath = '../Outputs/Logs/Conv3D_nC'+str(numClass)+'_LOS'+str(out_Session)+'_')

#%% test mode
model_cls_9.load_weights(modelsavefile) #model needs to be built first 
model_list.append(model_cls_9)

params_test = params.copy()
params_test['batch_size'] = 1
params_test['shuffle'] = False

test_gen = DataGenerator_mem(test_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params_test)
m_y_test = labels[test_list_ind]-1

m_y_pred = model_cls_9.predict(test_gen)
acc_test = sum(m_y_test == np.argmax(m_y_pred, axis=1)) / m_y_test.shape[0]

cm = confusion_matrix(m_y_test, np.argmax(m_y_pred, axis=1))
cm_list.append(cm)
acc_list.append(acc_test)
print(acc_test)
print(cm)
#%
tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path = out_path + 'Conv3D_' + \
        str(numClass) + '_LO' + str(out_Session)+'_')
#%% subcategory
numClass = 47
params = {'batch_size': 250, 'shuffle': True, 'n_classes': numClass}
params['dim'] = (128, 64, 50)
params['n_channels'] = 1
params['datapath'] = '../Data/SessionCSV/'
params['y_offset']=1  #first label is 1 instead of 0
params['labelpath']='../Data/labels_50_10/' # _50_5 or _50_10
params['labelmode']=str(numClass)
f=np.load(params['labelpath']+'LabelMeta'+str(numClass)+'.npz')
label_lens=f['arr_0']
Meta_Ind = f['arr_1'] # P, R, Slices, y-47, y-9
labels=Meta_Ind[:,3]  #3-47, 4-9
patience= 10

# load slice label and frame index data into memory
mSliceDict = {}
for P in range(1,plim):
    for R in range(1,4):
        datastr = 'P'+str(P)+'R'+str(R)
        mSliceDict[datastr]=np.genfromtxt(
            params['labelpath']+datastr+'_label_'+str(numClass)+'b.csv', delimiter=',',dtype=int)
        
for subcat in range( 1, 10 ): #subcategory  (1, 10)
    #% leave session out
    # select the training and testing indexes based on person, recording and class conditions
    train_list = np.where( (Meta_Ind[:,1]!=out_Session) & 
                          (Meta_Ind[:,0]<plim) & 
                          (Meta_Ind[:,4]==subcat))[0].tolist()
    test_list = np.where( (Meta_Ind[:,1]==out_Session)  & 
                         (Meta_Ind[:,0]<plim) & 
                         (Meta_Ind[:,4]==subcat))[0].tolist()
    
    test_subind, valid_subind = tools.train_valid_split_jump(np.array(test_list), 10)  #train_list
    train_list_ind = train_list
    valid_list_ind = np.array(test_list)[valid_subind].tolist()  # may not be enough when plim is too small
    test_list_ind = np.array(test_list)[test_subind].tolist()
    
    # for liminting classes of the subcategory
    unique_labels=np.unique(Meta_Ind[ np.where((Meta_Ind[:,4]==subcat)) ,3])
    numClass = len(unique_labels)
    params['n_classes']=numClass
    params['y_offset']=unique_labels.min()
    
    enc_layers_sub = tf.keras.models.clone_model( model_list[subcat-1].layers[1] )
    cls_layers_sub = build_conv_classifier(numClass, kernel=5, dropoutrate=0.2)
    m_input = keras.Input(shape = (128, 64 ,50, 1)) 
    m_output = enc_layers_sub(m_input)
    m_output = cls_layers_sub(m_output)
    
    m_ini_learning_rate = 0.0001
    m_opt = keras.optimizers.Adam(learning_rate=m_ini_learning_rate)
    
    model_cls_sub = keras.Model(
        inputs = m_input,
        outputs = m_output,
    )
    m_opt = keras.optimizers.Adam(learning_rate=m_ini_learning_rate)
    model_cls_sub.compile(optimizer=m_opt,
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model_cls_sub.summary()
    modelsavefile = '../Outputs/model_Conv3D'+'_sub'+str(subcat)+'.h5'
    
    train_generator = DataGenerator_mem(train_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict,**params)
    valid_generator = DataGenerator_mem(valid_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params)
    model_cls_sub, history = tools.train_gen(
        model_cls_sub, epoch, train_generator, valid_generator, modelsavefile, patience, Batch_size=params['batch_size'], 
        initial_learning_rate=m_ini_learning_rate, logpath = '../Outputs/Logs/Conv3D_sub'+str(subcat)+'_LOS'+str(out_Session)+'_')
    #% test
    model_cls_sub.load_weights(modelsavefile) #model needs to be built first 
    model_list.append(model_cls_sub)
    
    params_test = params.copy()
    params_test['batch_size'] = 1
    params_test['shuffle'] = False
    
    test_gen = DataGenerator_mem(test_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params_test)
    m_y_test = labels[test_list_ind]-params_test['y_offset']
    
    m_y_pred = model_cls_sub.predict(test_gen)
    acc_test = sum(m_y_test == np.argmax(m_y_pred, axis=1)) / m_y_test.shape[0]
    
    cm = confusion_matrix(m_y_test, np.argmax(m_y_pred, axis=1))
    cm_list.append(cm)
    acc_list.append(acc_test)
    print(acc_test)
    print(cm)
    #%
    tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path = out_path + 'Conv3D_sub'+str(subcat)+ '_LO' + str(out_Session)+'_')
#%%
numClass = 47
patience= 50
params['n_classes']=numClass
params['y_offset']=1  #first label is 1 instead of 0
train_list = np.where( (Meta_Ind[:,1]!=out_Session) & 
                      (Meta_Ind[:,0]<plim) )[0].tolist()
test_list = np.where( (Meta_Ind[:,1]==out_Session)  & 
                     (Meta_Ind[:,0]<plim) )[0].tolist()

test_subind, valid_subind = tools.train_valid_split_jump(np.array(test_list), 10)  #train_list
train_list_ind = train_list#np.array(train_list)[train_subind].tolist()
valid_list_ind = np.array(test_list)[valid_subind].tolist()  #train_list
test_list_ind = np.array(test_list)[test_subind].tolist()

for i in range(2,3): #3
    for j in range(0,1): #2
        print(str(i)+','+str(j))
        
        if i==0:
            enc_layers_47 = tf.keras.models.clone_model( model_list[9].layers[1] ) # use the latest model in the list
            for l in range(3):
                enc_layers_47.layers[l].trainable = False
        elif i==1:
            enc_layers_47 = tf.keras.models.clone_model( model_list[9].layers[1] ) # use the latest model in the list
        elif i==2:
            enc_layers_47 = build_conv_encoder()
            
        if j==0:
            #cls_layers_47 = build_conv_classifier(numClass, kernel=5, dropoutrate=0.2)
            cls_layers_47 = build_dense_classifier(numClass, kernel=5, dropoutrate=0.2)
            m_input = keras.Input(shape = (128, 64 ,50, 1)) 
            m_output = enc_layers_47(m_input)
            m_output = cls_layers_47(m_output)
        elif j==1:
            cls_layers_47 = keras.models.Sequential([        
                layers.Dropout(0.2),
                layers.Conv3D( filters = 1, kernel_size = 5, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Reshape((8,4,5)),
                layers.Conv2D( filters = 1, kernel_size = 5, padding='same', activation='relu'),
                layers.Flatten()
                ])
            enc_cls_9 = tf.keras.models.clone_model(model_list[0])
            enc_cls_9.trainable = False
            m_input = keras.Input(shape = (128, 64 ,50, 1)) 
            m_output_47 = enc_layers_47(m_input)
            m_output_47 = cls_layers_47(m_output_47)
            m_output_9 = enc_cls_9(m_input)
            m_output_47 = layers.concatenate([m_output_47, m_output_9])
            m_output = layers.Dense(numClass, activation='softmax')(m_output_47)
        
        model_cls_47 = keras.Model(
            inputs = m_input,
            outputs = m_output,
        )
        
        m_ini_learning_rate = 0.0001
        m_opt = keras.optimizers.Adam(learning_rate=m_ini_learning_rate)
        
        model_cls_47.compile(optimizer=m_opt,
                      loss=keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        model_cls_47.summary()
        
        modelsavefile = '../Outputs/model_Conv3D'+'_'+str(numClass)+'.h5'
        train_generator = DataGenerator_mem(train_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict,**params)
        valid_generator = DataGenerator_mem(valid_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params)
        model_cls_47, history = tools.train_gen(
            model_cls_47, epoch, train_generator, valid_generator, modelsavefile, patience, Batch_size=params['batch_size'], 
            initial_learning_rate=m_ini_learning_rate, logpath = '../Outputs/Logs/Conv3D_nC'+
            str(47)+'_LOS'+str(out_Session)+'_ij'+str(i)+str(j)+'_' )
        #% test mode
        model_cls_47.load_weights(modelsavefile) #model needs to be built first 
        model_list.append(model_cls_sub)
        params_test = params.copy()
        params_test['batch_size'] = 1
        params_test['shuffle'] = False
        
        test_gen = DataGenerator_mem(test_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params_test)
        m_y_test = labels[test_list_ind]-1
        
        m_y_pred = model_cls_47.predict(test_gen)
        acc_test = sum(m_y_test == np.argmax(m_y_pred, axis=1)) / m_y_test.shape[0]
        
        cm = confusion_matrix(m_y_test, np.argmax(m_y_pred, axis=1))
        cm_list.append(cm)
        acc_list.append(acc_test)
        print(acc_test)
        print(cm)
        #%
        tools.plot_confusion_matrix(cm, range(1, numClass+1), file_path = out_path + 'Conv3D_' + \
                str(numClass) + '_LO' + str(out_Session)+'_ij'+str(i)+str(j)+'_')
#%% save model, cm, acc lists
cm_all.append(cm_list)
acc_all.append(cm_list)
open_file = open('../Outputs/results-'+datetime.now().strftime("%Y%m%d-%H%M%S"), "wb")
#pickle.dump(model_list, open_file)
pickle.dump(cm_all, open_file)
pickle.dump(acc_all, open_file)
open_file.close()