# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 12:42:57 2022

@author: fredr
"""



import matplotlib.pyplot as plt
import data_parser as parser
import numpy as np
import toolbox as tools

y_pred = np.load('../Outputs/results/Conv3D_y_pred-01281225.npy')
y_test = np.load('../Outputs/results/Conv3D_y_test-01281225.npy')
test_list = np.load('../Outputs/results/Conv3D_test_list_ind-01281408.npy')


numClass = 47
f=np.load('../Data/labels_50_10/LabelMeta'+str(numClass)+'.npz')
label_lens=f['arr_0']
Meta_Ind = f['arr_1']

sample = 10 #which sample to take

figure = plt.figure(figsize=(34,20))
W = 12
H = 4
y_test = y_test+1
y_pred = y_pred+1
for A in range(1,48): #activity 1,48
    ind = np.where((y_pred != y_test) &
                    (y_test == A))[0]
    
    print(str(A) + ' ' + str(len(ind)))
    if len(ind)>0:
        clip, label = parser.clip_from_metaindex(ind[sample], '../Data/SessionCSV/', '../Data/labels_50_10/', str(numClass))
        clip=np.swapaxes(clip,0,1)
        sumFrame = np.max(clip, axis = 2)
        axis = plt.subplot(H, W, A)
        plt.imshow(sumFrame)
        plt.tight_layout()
        plt.subplots_adjust(hspace=.2, wspace=0.01)
        plt.title('True:'+str(y_test[ind[sample]]) + ', Pred:'+str(y_pred[ind[sample]]),y=-0.12)
plt.savefig('../Outputs/DataSamples/false_negatives.png')
plt.close()

sample = 20 #which sample to take

figure = plt.figure(figsize=(34,20))
W = 12
H = 4
for A in range(1,48): #activity 1,48
    ind = np.where((y_pred != y_test) &
                    (y_pred == A))[0]
    
    print(str(A) + ' ' + str(len(ind)))
    if len(ind)>0:
        clip, label = parser.clip_from_metaindex(ind[sample], '../Data/SessionCSV/', '../Data/labels_50_10/', str(numClass))
        clip=np.swapaxes(clip,0,1)
        sumFrame = np.max(clip, axis = 2)
        axis = plt.subplot(H, W, A)
        plt.imshow(sumFrame)
        plt.tight_layout()
        plt.subplots_adjust(hspace=.2, wspace=0.01)
        plt.title('True:'+str(y_test[ind[sample]]) + ', Pred:'+str(y_pred[ind[sample]]),y=-0.12)
plt.savefig('../Outputs/DataSamples/false_positives.png')
plt.close()