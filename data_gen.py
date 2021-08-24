# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:49:17 2021

@author: bzhou
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:11:59 2020

@author: zhoubo
"""
import numpy as np
import data_parser as parser
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=32, dim=(128,64,50), n_channels=1,
                 n_classes=47, shuffle=True, datapath='PY/datagen_2d/all/', labelpath=''):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        #self.labels = labels
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.datapath = datapath
        self.labelpath = labelpath
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))  # total data samples
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample and class
            tempX, y[i] = parser.clip_from_metaindex(ID, datapath=self.datapath, labelpath=self.labelpath, labelmode=str(self.n_classes))
            X[i,] = np.swapaxes(tempX[:, :, :, np.newaxis], 0, 1)
        return X, keras.utils.to_categorical(y-1, num_classes=self.n_classes)
    
    #%%
class DataGenerator_mem(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, datadict, Meta_Ind, slicedict, batch_size=32, dim=(128,64,50), n_channels=1,
                 n_classes=47, shuffle=True, datapath='PY/datagen_2d/all/', labelpath='', y_offset = 1, labelmode = ''):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.datadict = datadict
        self.Meta_Ind = Meta_Ind
        self.slicedict = slicedict
        #self.labels = labels
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.datapath = datapath
        self.labelpath = labelpath
        self.y_offset = y_offset
        if labelmode == '':
            self.labelmode = str(self.n_classes)
        else:
            self.labelmode = labelmode
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))  # total data samples
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample and class
            tempX, y[i] = parser.clip_from_metaindex_mem(ID, datadict=self.datadict, Meta_Ind=self.Meta_Ind, slicedict = self.slicedict,
                                                         datapath=self.datapath, labelpath=self.labelpath, labelmode=self.labelmode)
            X[i,] = np.swapaxes(tempX[:, :, :, np.newaxis], 0, 1)
        return X, keras.utils.to_categorical(y-self.y_offset, num_classes=self.n_classes)
    