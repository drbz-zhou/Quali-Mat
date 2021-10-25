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
def build_conv_classifier(kernel=5, dropoutrate=0.2, numClass=47):
    model = keras.models.Sequential([        
        layers.Dropout(dropoutrate),
        layers.Conv3D( filters = 1, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Reshape((8,4,5)),
        layers.Conv2D( filters = 1, kernel_size = kernel, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(numClass, activation='softmax')
        ])
    return model

encoder = build_conv_encoder()
decoder = build_conv_decoder()
m_input = keras.Input(shape = (128, 64 ,50, 1)) 
m_output = encoder(m_input)
m_output = decoder(m_output)
model_aec = keras.Model(
    inputs = m_input,
    outputs = m_output,
)

m_ini_learning_rate = 0.001
m_opt = keras.optimizers.Adam(learning_rate=m_ini_learning_rate)
model_aec.compile(optimizer=m_opt,
              loss=keras.losses.MeanSquaredError(),
              metrics = keras.metrics.MeanSquaredError())
model_aec.summary()
encoder.summary()
decoder.summary()


#%% setup GPU
tools.tf_setup_GPU()
#tools.tf_mem_patch()
numClass = 47 #9, 47
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
epoch = 20
modelsavefile = '../Outputs/model_ConvAutoEncoder.h5'

#%
autoencoder_gen = DataGenerator_mem_reconstruct(train_list_ind+valid_list_ind+test_list_ind, 
                                                datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict,**params)
#valid_gen = DataGenerator_mem_reconstruct(valid_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params)
# train model
history = model_aec.fit( x = autoencoder_gen, epochs = epoch, batch_size=params['batch_size'],
              #use_multiprocessing = True,
              #validation_data = m_datagen_valid,
              #callbacks=[cb_checkpoint, cb_earlystop, cb_learningrate, cb_tensorboard], #, cb_learningrate
              verbose = 1 # 2 if log to file
        )

#%% build the classification model

for model_arch in ["Conv3D_vanilla", "Conv3D_aec_unlocked", "Conv3D_aec_diffLR"]:
    #"Conv3D_vanilla", "Conv3D_aec_unlocked", "Conv3D_aec_locked", "Conv3D_aec_finetune", "Conv3D_aec_diffLR"]:
    if model_arch == "Conv3D_aec_locked":
        enc_layers = tf.keras.models.clone_model( model_aec.layers[1] )
        enc_layers.trainable = False
    elif model_arch == "Conv3D_aec_unlocked":
        enc_layers = tf.keras.models.clone_model( model_aec.layers[1] )
        enc_layers.trainable = True
    elif model_arch == "Conv3D_aec_finetune":
        enc_layers = tf.keras.models.clone_model( model_aec.layers[1] )
        enc_layers.trainable = True
        fine_tune_at = 8
        for layer in enc_layers.layers[:fine_tune_at]:  # in total 16 layers
            layer.trainable = False
        model_arch = model_arch+"_"+str(fine_tune_at)
    elif model_arch == "Conv3D_vanilla":
        enc_layers = build_conv_encoder()
    elif model_arch == "Conv3D_aec_diffLR":
        enc_layers = tf.keras.models.clone_model( model_aec.layers[1] )
        
        
    cls_layers = build_conv_classifier(kernel=5, dropoutrate=0.2)
    
    enc_layers.summary()
    m_input = keras.Input(shape = (128, 64 ,50, 1)) 
    m_output = enc_layers(m_input)
    m_output = cls_layers(m_output)
    
    model_cls = keras.Model(
        inputs = m_input,
        outputs = m_output,
    )
    
    if model_arch == "Conv3D_aec_diffLR":
        var_list_enc = enc_layers.trainable_variables
        var_list_cls = cls_layers.trainable_variables
        optimizers = [
            tf.keras.optimizers.Adam(learning_rate=0.00002),
            tf.keras.optimizers.Adam(learning_rate=0.0001)
        ]
        optimizers_and_layers = [(optimizers[0], model_cls.layers[:2]), (optimizers[1], model_cls.layers[2])]
        m_opt = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
    else:    
        m_ini_learning_rate = 0.0001
        m_opt = keras.optimizers.Adam(learning_rate=m_ini_learning_rate)
    
    model_cls.compile(optimizer=m_opt,
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model_cls.summary()
    
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    epoch = 100
    modelsavefile = '../Outputs/model_'+model_arch+'_'+str(numClass)+'.h5'
    patience= 100
    
    #%
    train_generator = DataGenerator_mem(train_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict,**params)
    valid_generator = DataGenerator_mem(valid_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params)
    # train model
    if model_arch == "Conv3D_aec_diffLR":
        model_cls, history = tools.train_gen_nodecay(
            model_cls, epoch, train_generator, valid_generator, modelsavefile, patience, Batch_size=params['batch_size'], 
            initial_learning_rate=m_ini_learning_rate, logpath = '../Outputs/Logs/'+model_arch+'_LOS'+str(out_Session)+'_', weights_only=True)
    else:
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