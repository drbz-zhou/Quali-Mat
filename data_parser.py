# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 11:07:49 2021

@author: bzhou
"""

import numpy as np
import NSMutils as tools

def read_frame(filename, index, lineindex):
    indexlist = np.genfromtxt(lineindex,delimiter=',',dtype=np.longlong)
    f = open(filename)
    f.seek(indexlist[index])                  #go to random position
    #f.readline()                    # discard - bound to be partial line
    mLineStr = f.readline()      # bingo!
    # # extra to handle last/first line edge cases
    # if len(mLineStr) == 0:       # we have hit the end
    #     f.seek(index)
    #     mLineStr = f.readline()  # so we'll grab the first line instead
    #     print('EOF')
    mLineArray = np.fromstring(mLineStr, sep=',')
    mFrame = mLineArray.reshape((64,128))
    return mFrame


def get_clip(filename, index_start, index_end, lineindex):
    mClip = np.zeros( (64,128, int(index_end-index_start)))
    indexlist = np.genfromtxt(lineindex,delimiter=',',dtype=np.longlong)
    f = open(filename)
    f.seek(indexlist[index_start]) 
    for i in range( int(index_start), int(index_end)):
        mLineStr = f.readline()
        mLineArray = np.fromstring(mLineStr, sep=',')
        mFrame = mLineArray.reshape((64,128))
        mClip[:,:,i-int(index_start)] = mFrame
    f.close()
    return mClip

def get_clip_np(filepath, index_start, index_end, chunksize=100):
    mClip = np.zeros( (64,128, int(index_end-index_start)))
    chunk_start = np.floor_divide(int(index_start),int(chunksize))
    chunk_end = np.floor_divide(int(index_end),int(chunksize))
    chunk_index_start = np.mod(int(index_start),int(chunksize))
    chunk_index_end = np.mod(int(index_end),int(chunksize))
    if chunk_start==chunk_end:
        chunkdata = np.load(filepath+str(chunk_start)+'.npy')
        mClip = chunkdata[:,:,chunk_index_start:chunk_index_end]
    else:
        chunkdata1 = np.load(filepath+str(chunk_start)+'.npy')
        chunkdata2 = np.load(filepath+str(chunk_end)+'.npy')
        mClip = np.concatenate( (chunkdata1[:,:,chunk_index_start:chunksize],
                                 chunkdata2[:,:,0:chunk_index_end]), axis=2 )
    return mClip

def get_clip_np_mem(mData, index_start, index_end):
    mClip = mData[:,:,index_start:index_end]
    return mClip

def decode_index(i,labelpath, labelmode): #='G:/NewSmartMat/Data/labels/'
    filename_meta = 'LabelMeta'+labelmode+'.npz'
    f = np.load(labelpath+filename_meta)
    Label_Lens = f['arr_0']
    Meta_Ind = f['arr_1']
    [P,R,S]=Meta_Ind[i,0:3]
    return P, R, S

def decode_index_mem(i,Meta_Ind): #='G:/NewSmartMat/Data/labels/'
    [P,R,S]=Meta_Ind[i,0:3]
    return P, R, S

def slice_from_PRS(P, R, S, datapath, labelpath, labelmode): #='Data/'  ='G:/NewSmartMat/Data/labels/'
    filename_data = datapath + '/P'+str(P)+'R'+str(R)+'.csv'
    filename_label = 'P'+str(P)+'R'+str(R)+'_label_'+labelmode+'b.csv'
    filename_meta = 'LabelMeta'+labelmode+'.npz'
    #get index_start and index_end from label file
    
    slices = np.genfromtxt(labelpath+filename_label, delimiter=',',dtype=int)
    index_start=slices[S,0]
    index_end = slices[S,1]
    if labelmode == '47':
        label = slices[S,2]
    elif labelmode == '9':
        label = slices[S,3]
    return int(index_start), int(index_end), int(label), filename_data

def slice_from_PRS_mem(S, slices, labelmode): #='Data/'  ='G:/NewSmartMat/Data/labels/'
    index_start=slices[S,0]
    index_end = slices[S,1]
    if labelmode == '47':
        label = slices[S,2]
    elif labelmode == '9':
        label = slices[S,3]
    return int(index_start), int(index_end), int(label)

def clip_from_metaindex(i, datapath, labelpath, labelmode): # = 'Data/' ='G:/NewSmartMat/Data/labels/'
    P, R, S = decode_index(i,labelpath, labelmode)
    index_start, index_end, label, filename_data = slice_from_PRS(P, R, S,datapath,labelpath, labelmode)
    linindex = datapath + 'P'+str(P)+'R'+str(R)+'_LineInd.csv'
    #clip = get_clip(filename_data, index_start, index_end, linindex)
    filepath = datapath + 'P'+str(P)+'R'+str(R)+'Chunks/'
    
    clip = get_clip_np(filepath, index_start, index_end)
    return clip, label

def clip_from_metaindex_mem(i, datadict, Meta_Ind, slicedict, datapath, labelpath, labelmode): # = 'Data/' ='G:/NewSmartMat/Data/labels/'
    P, R, S = decode_index_mem(i,Meta_Ind)
    datastr = 'P'+str(P)+'R'+str(R)
    index_start, index_end, label = slice_from_PRS_mem(S,slicedict[datastr], labelmode)
    clip = get_clip_np_mem(datadict[datastr], index_start, index_end)
    return clip, label

    # mLineArray = np.genfromtxt(rootpath+str(index)+'.csv', delimiter=',')
    # mFrame = mLineArray.reshape((64,128))
    # return mFrame
    
# def get_clip(filename, index_start, index_end, lineindex):
#     mClip = np.zeros( (64,128, int(index_end-index_start)))
#     for i in range( int(index_start), int(index_end)):
#         mClip[:,:,i-int(index_start)] = read_frame(filename, i, lineindex)
#     return mClip