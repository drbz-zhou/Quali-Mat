# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 22:41:58 2020

@author: MyPC
"""

import numpy as np
import pickle
import itertools
import matplotlib.pyplot as plt
from tensorflow import keras
from datetime import datetime
import logging
import subprocess, os, sys
import tensorflow as tf
import math

def tf_board():
    return

def tf_setup_GPU(ind=0):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.experimental.set_visible_devices(gpus[ind], 'GPU')
        print('running on GPU:'+str(gpus[ind]))
      except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)
    return

def tf_mem_patch():
    # to cure tensorflow memory allocation problem
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    # end curing memory allocation problem
    return session

def tf_setlogger(filepath = 'Outputs/Logs/'):
    
    # # get TF logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)
    
    # # create formatter and add it to the handlers
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # # create file handler which logs even debug messages
    now = datetime.now()
    date_time = now.strftime("%m%d-%H%M%S")
    logfilename = filepath+'log-'+date_time+'.log'
    fh = logging.FileHandler(logfilename)
    fh.setLevel(logging.INFO)
    #fh.setFormatter(formatter)
    logger.addHandler(fh)

    return fh, logger

def load_data_valid(leaveoutsession, bins):
    #1,2,3 sessinos,  1,2,3,4 bins
    m_data_valid = np.zeros( (0, 128, 64, 50, 1) )
    m_labels0_valid = np.zeros( (0) )
    m_labels1_valid = np.zeros( (0) )
    m_labelsP_valid = np.zeros( (0) )
    m_labelsR_valid = np.zeros( (0) )
    r = leaveoutsession
    for b in bins:
        m_filename = "PY/window_data/Sess"+str(r)+"_bin"+str(b)+".txt"
        print(m_filename)
        with open(m_filename,"rb") as fp:   #read file
            m_datlist = pickle.load(fp)
            m_labels0 = pickle.load(fp)
            m_labels1 = pickle.load(fp)
            m_labelsP = pickle.load(fp)
            m_labelsR = pickle.load(fp)
            fp.close
        print( [m_datlist[0].shape, len(m_datlist), len(m_labels0)] )
        m_data_valid = np.concatenate( ( m_data_valid, np.array(m_datlist) ) )
        m_labels0_valid = np.concatenate( ( m_labels0_valid, np.array(m_labels0) ) )
        m_labels1_valid = np.concatenate( ( m_labels1_valid, np.array(m_labels1) ) )
        m_labelsP_valid = np.concatenate( ( m_labelsP_valid, np.array(m_labelsP) ) )
        m_labelsR_valid = np.concatenate( ( m_labelsR_valid, np.array(m_labelsR) ) )
        del m_datlist, m_labels0, m_labels1
    return m_data_valid, m_labels0_valid, m_labels1_valid, m_labelsP_valid, m_labelsR_valid
    

def load_data_train(leaveoutsession, bins):
    m_data_train = np.zeros( (0, 128, 64, 50, 1) )
    m_labels0_train = np.zeros( (0) )
    m_labels1_train = np.zeros( (0) )
    for r in range (1,4):
        if r!=leaveoutsession:
            for b in bins:
                m_filename = "PY/window_data/Sess"+str(r)+"_bin"+str(b)+".txt"
                print(m_filename)
                with open(m_filename,"rb") as fp:   #read file
                    m_datlist = pickle.load(fp)
                    m_labels0 = pickle.load(fp)
                    m_labels1 = pickle.load(fp)
                    fp.close
                print( [m_datlist[0].shape, len(m_datlist), len(m_labels0)] )
                m_data_train = np.concatenate( ( m_data_train, np.array(m_datlist) ) )
                m_labels0_train = np.concatenate( ( m_labels0_train, np.array(m_labels0) ) )
                m_labels1_train = np.concatenate( ( m_labels1_train, np.array(m_labels1) ) )
                del m_datlist, m_labels0, m_labels1
    return m_data_train, m_labels0_train, m_labels1_train


def label2categorical(m_labels, numClass):
    m_labels_exp = np.zeros( ( len(m_labels), numClass ) )
    for i in range (0, len(m_labels)):
        m_labels_exp[i, int(m_labels[i])-1] = 1
    return m_labels_exp


def train_step(model, epoch, m_data_train, m_y_train, m_data_valid, m_y_valid, modelsavefile, Patience = 50):
    cb_checkpoint = keras.callbacks.ModelCheckpoint(modelsavefile, monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    cb_earlystop = keras.callbacks.EarlyStopping(patience=Patience, monitor='val_accuracy', verbose = 1, restore_best_weights=True )
    history = model.fit( x = m_data_train, y = m_y_train, epochs = epoch, batch_size=50,
              use_multiprocessing = True,
              validation_data = (m_data_valid, m_y_valid),
              callbacks=[cb_checkpoint, cb_earlystop],
              #callbacks=[cb_earlystop],  #sometimes can't save model because of h5 bug, early stop restore best weights
              verbose = 1 # 2 if log to file
        )
    return model, history

def lr_time_based_decay(epoch, lr, initial_learning_rate):
    decay = initial_learning_rate / epoch
    return lr * 1 / (1 + decay * epoch)

def lr_step_decay(epoch, lr, initial_learning_rate):
    drop_rate = 0.5
    epochs_drop = 10.0
    return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))

def lr_scheduler_exp(epoch, lr):
    if epoch < 50:
        return lr
    elif epoch % 10 != 1:
        return lr
    else:
        return lr * 0.5 #* math.exp(-0.1)
    
def train_gen(model, epoch, m_datagen_train, m_datagen_valid, modelsavefile, 
              Patience = 50, Batch_size = 32, weights_only=False, initial_learning_rate=0.001, logpath='../Outputs/Logs/'):
    cb_learningrate=keras.callbacks.LearningRateScheduler(lr_scheduler_exp, verbose=1)
    cb_checkpoint = keras.callbacks.ModelCheckpoint(modelsavefile, monitor='val_accuracy', mode='max', 
                                                    verbose=1, save_weights_only=weights_only,save_best_only=True)
    cb_earlystop = keras.callbacks.EarlyStopping(patience=Patience, monitor='val_accuracy', verbose = 1, 
                                                 restore_best_weights=True )    
    log_dir = logpath+ datetime.now().strftime("%Y%m%d-%H%M%S")
    cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit( x = m_datagen_train, epochs = epoch, batch_size=Batch_size,
              #use_multiprocessing = True,
              validation_data = m_datagen_valid,
              callbacks=[cb_checkpoint, cb_earlystop, cb_learningrate, cb_tensorboard], #, cb_learningrate
              verbose = 1 # 2 if log to file
        )
    return model, history

def train_gen_nodecay(model, epoch, m_datagen_train, m_datagen_valid, modelsavefile, 
              Patience = 50, Batch_size = 32, weights_only=False, initial_learning_rate=0.001, logpath='../Outputs/Logs/'):
    cb_checkpoint = keras.callbacks.ModelCheckpoint(modelsavefile, monitor='val_accuracy', mode='max', 
                                                    verbose=1, save_weights_only=weights_only,save_best_only=True)
    cb_earlystop = keras.callbacks.EarlyStopping(patience=Patience, monitor='val_accuracy', verbose = 1, 
                                                 restore_best_weights=True )    
    log_dir = logpath+ datetime.now().strftime("%Y%m%d-%H%M%S")
    cb_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit( x = m_datagen_train, epochs = epoch, batch_size=Batch_size,
              #use_multiprocessing = True,
              validation_data = m_datagen_valid,
              callbacks=[cb_checkpoint, cb_earlystop, cb_tensorboard], #, cb_learningrate
              verbose = 1 # 2 if log to file
        )
    return model, history

def train_gen_autovalid(model, epoch, m_datagen_train, modelsavefile, Patience = 50, Batch_size = 32, weights_only=False):
    cb_checkpoint = keras.callbacks.ModelCheckpoint(modelsavefile, monitor='val_accuracy', mode='max', 
                                                    verbose=1, save_weights_only=weights_only,save_best_only=True)
    cb_earlystop = keras.callbacks.EarlyStopping(patience=Patience, monitor='val_accuracy', verbose = 1, restore_best_weights=True )
    history = model.fit( x = m_datagen_train, epochs = epoch, batch_size=Batch_size,
              #use_multiprocessing = True,
              validation_split = 0.1,
              callbacks=[cb_checkpoint, cb_earlystop],
              verbose = 1
        )
    return model, history

def train_test_gen(model, epoch, m_datagen_train, m_datagen_valid, m_datagen_test, m_y_test, modelsavefile, Patience = 50, Batch_size = 32):
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    test_acc = []
    max_val_acc = 0
    for e in range(epoch):
        history = model.fit( x = m_datagen_train, epochs = 1, batch_size=Batch_size,
                  use_multiprocessing = True,
                  validation_data = m_datagen_valid,
                  #callbacks=[cb_checkpoint, cb_earlystop],
                  #callbacks=[cb_earlystop],  #sometimes can't save model because of h5 bug, early stop restore best weights
                  verbose = 1
            )
        acc, val_acc, loss, val_loss = append_history(history, acc, val_acc, loss, val_loss)
        m_y_pred = model.predict(m_datagen_test)
        test_acc_epoch = sum(m_y_test == np.argmax(m_y_pred, axis=1)) / m_y_test.shape[0]
        test_acc = np.concatenate( (test_acc, test_acc_epoch) )
        if val_acc > max_val_acc:
            print(['validation accuracy improved from '+str(max_val_acc) +' to '+str(val_acc)])
            model.save(modelsavefile)
            max_val_acc = val_acc
    return model, acc, val_acc, loss, val_loss, test_acc

def append_history(history, acc, val_acc, loss, val_loss):
    acc = np.concatenate( ( acc, np.array(history.history['accuracy'])))
    val_acc = np.concatenate( ( val_acc, np.array(history.history['val_accuracy'])))
    loss = np.concatenate( ( loss, np.array(history.history['loss'])))
    val_loss = np.concatenate( ( val_loss, np.array(history.history['val_loss'])))
    return acc, val_acc, loss, val_loss

def plot_confusion_matrix(cm, class_names, if_save = True, file_path = '', title_prefix='Confusion matrix'):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(len(class_names)/2, len(class_names)/2))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    TP = 0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        if i==j:
            TP=TP+cm[i,i]
    Acc = TP/cm.sum()
    plt.title(title_prefix+" acc:"+'%.2f' % (Acc*100)+"%")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.show()
    if if_save:
        now = datetime.now()
        date_time = now.strftime("%m%d%H%M")
        plt.savefig(file_path+'CM-'+date_time+'.png')
        np.save(file_path+'CM-'+date_time+'.npy', cm)
    else:
        plt.show()
    return figure


def plot_acc_loss(acc, val_acc, loss, val_loss, if_save = True, file_path = ''):
    epochs_range = range(len(acc))
    
    figure = plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    #plt.show()
    if if_save:
        now = datetime.now()
        date_time = now.strftime("%m%d%H%M")
        plt.savefig(file_path+'AccLoss-'+date_time+'.png')
    else:
        plt.show()
    return figure

def print_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

def balance_randomsample(data, labels, dims = 2):
    labels_unq = np.unique(labels)
    numClass = labels_unq.shape[0]
    m_counts = np.zeros( (numClass,1) )
    for c in range(numClass):
        m_counts[c] = np.where(labels == labels_unq[c])[0].shape[0]
    labels_new = labels
    data_new = data
    for c in range(numClass):
        print(c)
        m_delta = int(np.max(m_counts)-m_counts[c])
        ind = np.random.randint(0,m_counts[c], (m_delta,))
        ind_in_data = np.where(labels == labels_unq[c])[0][ind]
        labels_new = np.concatenate( (labels_new, labels[ind_in_data]) ) 
        if dims == 2:
            data_new  = np.concatenate( (data_new, data[ind_in_data,:,:]) ) 
        elif dims == 3:
            data_new  = np.concatenate( (data_new, data[ind_in_data,:,:, :]) ) 
    return data_new, labels_new

def balance_randomsample2(data, labels0, labels1, dims = 2):
    #labels0: 47, labels1: 9, balance according to labels0
    labels0_unq = np.unique(labels0)
    numClass = labels0_unq.shape[0]
    m_counts = np.zeros( (numClass,1) )
    for c in range(numClass):
        m_counts[c] = np.where(labels0 == labels0_unq[c])[0].shape[0]
    labels0_new = labels0
    labels1_new = labels1
    data_new = data
    for c in range(numClass):
        print(c)
        m_delta = int(np.max(m_counts)-m_counts[c])
        ind = np.random.randint(0,m_counts[c], (m_delta,))
        ind_in_data = np.where(labels0 == labels0_unq[c])[0][ind]
        labels0_new = np.concatenate( (labels0_new, labels0[ind_in_data]) ) 
        labels1_new = np.concatenate( (labels1_new, labels1[ind_in_data]) ) 
        if dims == 2:
            data_new  = np.concatenate( (data_new, data[ind_in_data,:,:]) ) 
        elif dims == 3:
            data_new  = np.concatenate( (data_new, data[ind_in_data,:,:, :]) ) 
    return data_new, labels0_new, labels1_new

def balance_randomsample_all_labels(data, labels0, labels1, labelsP, labelsR, dims = 2):
    #labels0: 47, labels1: 9, balance according to labels0
    labels0_unq = np.unique(labels0)
    numClass = labels0_unq.shape[0]
    m_counts = np.zeros( (numClass,1) )
    for c in range(numClass):
        m_counts[c] = np.where(labels0 == labels0_unq[c])[0].shape[0]
    labels0_new = labels0
    labels1_new = labels1
    labelsP_new = labelsP
    labelsR_new = labelsR
    data_new = data
    for c in range(numClass):
        print(c)
        # the difference between current class and most counted class
        m_delta = int(np.max(m_counts)-m_counts[c])  
        # generate random index to fill the gap
        ind = np.random.randint(0,m_counts[c], (m_delta,))  
        ind_in_data = np.where(labels0 == labels0_unq[c])[0][ind]
        labels0_new = np.concatenate( (labels0_new, labels0[ind_in_data]) ) 
        labels1_new = np.concatenate( (labels1_new, labels1[ind_in_data]) ) 
        labelsP_new = np.concatenate( (labelsP_new, labelsP[ind_in_data]) ) 
        labelsR_new = np.concatenate( (labelsR_new, labelsR[ind_in_data]) ) 
        if dims == 2:
            data_new  = np.concatenate( (data_new, data[ind_in_data,:,:]) ) 
        elif dims == 3:
            data_new  = np.concatenate( (data_new, data[ind_in_data,:,:, :]) ) 
    return data_new, labels0_new, labels1_new, labelsP_new, labelsR_new

def train_valid_split_jump(source_list, m_ratio):
    #randperm_subind = np.random.permutation(len(source_list))
    valid_subind = np.arange(0, len(source_list),m_ratio, dtype = int)
    train_subind = np.arange(1, len(source_list),m_ratio, dtype = int)
    for i in range(m_ratio-2):
        train_subind = np.concatenate( (train_subind, np.arange(i+2, len(source_list),m_ratio)) )
    return train_subind, valid_subind

def train_valid_split(source_list, m_ratio):
    randperm_subind = np.random.permutation(source_list)
    valid_subind = randperm_subind[0:np.floor_divide(len(source_list),m_ratio)]
    train_subind = randperm_subind[np.floor_divide(len(source_list),m_ratio):len(source_list)]
    #valid_subind = source_list[0:np.floor_divide(len(source_list),m_ratio)]
    #train_subind = source_list[np.floor_divide(len(source_list),m_ratio):len(source_list)]
    return train_subind, valid_subind

def save_history(file_path, acc, val_acc, loss, val_loss, cm):
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H-%M-%S")
    with open(file_path+"history-"+date_time+'.txt', "wb") as fp: 
        pickle.dump(acc, fp)
        pickle.dump(val_acc, fp)
        pickle.dump(loss, fp)
        pickle.dump(val_loss, fp)
        pickle.dump(cm, fp)
        fp.close()
        
def load_history(file_path):
    with open(file_path, "rb") as fp:
        acc = pickle.load(fp)
        val_acc = pickle.load(fp)
        loss = pickle.load(fp)
        val_loss = pickle.load(fp)
        cm = pickle.load(fp)
        fp.close()
    return acc, val_acc, loss, val_loss, cm

def cm2acc(cm):
    tp=0
    for i in range(cm.shape[0]):
        tp+=cm[i,i]
    return tp/(sum(sum(cm)))