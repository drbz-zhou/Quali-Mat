# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 09:37:42 2021

@author: bzhou
"""

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
import matplotlib.pyplot as plt
 
# setup GPU
tools.tf_setup_GPU()
#tools.tf_mem_patch()
numClass = 47
out_Session = 3
plim = 13 # 7 for 6 people (64GB RAM), 13 for all 12 people (needs 128GB RAM), 2 for 1 person with quick test

params = {'batch_size': 256, 'shuffle': True, 'n_classes': numClass}
params['dim'] = (128, 64, 50)
params['n_channels'] = 1
params['datapath'] = '../Data/SessionCSV/'
params['y_offset']=1  #first label is 1 instead of 0
params['labelpath']='../Data/labels_50_10/' # _50_5 or _50_10
params['labelmode']=str(numClass)
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
#%%
for subcat in range( 1, 10 ): #subcategory  (1, 10)
    #% leave session out
    # select the training and testing indexes based on person, recording and class conditions
    #train_list = np.where( (Meta_Ind[:,1]!=out_Session) & (Meta_Ind[:,0]<plim) )[0].tolist()
    #test_list = np.where( (Meta_Ind[:,1]==out_Session)  & (Meta_Ind[:,0]<plim) )[0].tolist()
    train_list = np.where( (Meta_Ind[:,1]!=out_Session) & 
                          (Meta_Ind[:,0]<plim) & 
                          (Meta_Ind[:,4]==subcat))[0].tolist()
    test_list = np.where( (Meta_Ind[:,1]==out_Session)  & 
                         (Meta_Ind[:,0]<plim) & 
                         (Meta_Ind[:,4]==subcat))[0].tolist()
    
    test_subind, valid_subind = tools.train_valid_split_jump(np.array(test_list), 10)  #train_list
    train_list_ind = train_list#np.array(train_list)[train_subind].tolist()
    valid_list_ind = np.array(test_list)[valid_subind].tolist()  #train_list
    test_list_ind = np.array(test_list)[test_subind].tolist()
    
    # for liminting classes of the subcategory
    unique_labels=np.unique(Meta_Ind[ np.where((Meta_Ind[:,4]==subcat)) ,3])
    numClass = len(unique_labels)
    params['n_classes']=numClass
    params['y_offset']=unique_labels.min()
        
    tools.print_time()
    
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
    elif model_arch=="Conv_Trans":
        model = model_builder.build_Conv_Trans(num_heads = 8, dff = 32, numClass = numClass, d_model = 32,
                         dropoutrate = 0.2, conv_filters = 5, conv_kernel = 3)
    elif model_arch=="Conv_Trans_w9":  #with model 9 as the category restrictor, numClass should be 47
        model = model_builder.build_Conv_Trans_w9(num_heads = 8, dff = 32, numClass = numClass, d_model = 32,
                         dropoutrate = 0.2, conv_filters = 5, conv_kernel = 3)
        
    m_opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=m_opt,
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.summary()
    
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    epoch = 10000
    modelsavefile = '../Outputs/model_'+model_arch+'_'+'subcat_'+str(subcat)+'.h5'
    patience= 50
    
    train_gen = DataGenerator_mem(train_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict,**params)
    valid_gen = DataGenerator_mem(valid_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params)
    # train model
    model, history = tools.train_gen(
        model, epoch, train_gen, valid_gen, modelsavefile, patience, Batch_size=params['batch_size'])
    acc, val_acc, loss, val_loss = tools.append_history(
        history, acc, val_acc, loss, val_loss)
    
    tools.print_time()
    #% test model
    model.load_weights(modelsavefile) #model needs to be built first 
    params_test = params.copy()
    params_test['batch_size'] = 1
    params_test['shuffle'] = False
                
    test_gen = DataGenerator_mem(test_list_ind, datadict=mDataDict, Meta_Ind=Meta_Ind, slicedict=mSliceDict, **params_test)
    m_y_test = labels[test_list_ind]-params_test['y_offset']
    
    m_y_pred = model.predict(test_gen)
    acc_test = sum(m_y_test == np.argmax(m_y_pred, axis=1)) / m_y_test.shape[0]
    
    cm = confusion_matrix(m_y_test, np.argmax(m_y_pred, axis=1))
    print('subcat_'+str(subcat))
    print(acc_test)
    print(cm)
    #%
    tools.plot_confusion_matrix(cm, range(params['y_offset'], numClass+params['y_offset']), file_path = out_path + model_arch + '_' + \
            str(numClass) + '_LO' + str(out_Session)+'_'+'subcat_'+str(subcat))
    tools.plot_acc_loss(acc, val_acc, loss, val_loss, file_path = out_path + model_arch + '_' + \
            str(numClass) + '_LO' + str(out_Session)+'_'+'subcat_'+str(subcat))
    
    #% plot failure examples
    
    figure = plt.figure(figsize=(32,20))
    W = 10
    H = 4
    y_test = m_y_test+params['y_offset']
    y_pred = np.argmax(m_y_pred, axis=1)+params['y_offset']
    sample = 0
    person = 1
    for i in range(len(unique_labels)): #activity 1,48
        A=unique_labels[i]
        ind = np.where((y_pred != y_test) &
                        (y_pred == A) &
                        (Meta_Ind[np.array(test_list_ind)][:, 0] == person))[0]      #change persons  
        print(str(A) + ' ' + str(len(ind)))
        person = person+1
        if len(ind)<1:
            ind = np.where((y_pred != y_test) &
                        (y_pred == A) )[0] 
        if len(ind)>sample:
            clip, label = parser.clip_from_metaindex_mem(ind[sample], mDataDict, Meta_Ind, mSliceDict, '../Data/SessionCSV/', '../Data/labels_50_10/', str(47))
            print(Meta_Ind[np.array(test_list_ind)][ind[sample], :])
        if len(ind)>10:
            clip, label = parser.clip_from_metaindex_mem(ind[10], mDataDict, Meta_Ind, mSliceDict, '../Data/SessionCSV/', '../Data/labels_50_10/', str(47))
            print(Meta_Ind[np.array(test_list_ind)][ind[10], :])
        if len(ind)>0:
            clip=np.swapaxes(clip,0,1)
            sumFrame = np.max(clip, axis = 2)
            axis = plt.subplot(H, W, i+1)
            plt.imshow(sumFrame)
            plt.tight_layout()
            plt.subplots_adjust(hspace=.2, wspace=0.01)
            plt.title('True:'+str(y_test[ind[sample]]) + ', Pred:'+str(y_pred[ind[sample]]) + ', ID'+str(Meta_Ind[np.array(test_list_ind[ind[sample]]), 0]),y=-0.12)
            print(Meta_Ind[np.array(test_list_ind[ind[sample]]), 0])
    for i in range(len(unique_labels)): #activity 1,48
        A=unique_labels[i]
        ind = np.where((y_pred != y_test) &
                        (y_test == A) &
                        (Meta_Ind[np.array(test_list_ind)][:, 0] == person))[0]      #change persons  
        print(str(A) + ' ' + str(len(ind)))
        if person == 12:
            person = 0
        person = person+1
        if len(ind)<1:
            ind = np.where((y_pred != y_test) &
                        (y_test == A) )[0] 
        if len(ind)>sample:
            clip, label = parser.clip_from_metaindex_mem(ind[sample], mDataDict, Meta_Ind, mSliceDict, '../Data/SessionCSV/', '../Data/labels_50_10/', str(47))
            print(Meta_Ind[np.array(test_list_ind)][ind[sample], :])
        if len(ind)>10:
            clip, label = parser.clip_from_metaindex_mem(ind[10], mDataDict, Meta_Ind, mSliceDict, '../Data/SessionCSV/', '../Data/labels_50_10/', str(47))
            print(Meta_Ind[np.array(test_list_ind)][ind[10], :])
        if len(ind)>0:
            clip=np.swapaxes(clip,0,1)
            sumFrame = np.max(clip, axis = 2)
            axis = plt.subplot(H, W, i+W+1)
            plt.imshow(sumFrame)
            plt.tight_layout()
            plt.subplots_adjust(hspace=.2, wspace=0.01)
            plt.title('True:'+str(y_test[ind[sample]]) + ', Pred:'+str(y_pred[ind[sample]]) + ', ID'+str(Meta_Ind[np.array(test_list_ind[ind[sample]]), 0]),y=-0.12)
            print(Meta_Ind[np.array(test_list_ind[ind[sample]]), 0])
    plt.savefig('../Outputs/DataSamples/false_examples_cat_'+str(subcat)+'.png')