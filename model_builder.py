# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 18:24:16 2021

@author: bzhou
"""
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import builder_transformer as trans_builder

def build_Conv3D(filters = 5, kernel = 3, dense = 256, numClass = 9, dropoutrate = 0.2):
    model = keras.models.Sequential([
        layers.Conv3D( filters = 10, kernel_size = kernel, padding='same', activation='relu', input_shape=(128, 64 ,50, 1)),
        layers.AveragePooling3D(pool_size=(2, 2, 3)),
        #layers.MaxPooling3D(pool_size=(2, 2, 3)),
        layers.Dropout(dropoutrate),
        layers.Conv3D( filters = 20, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(2, 2, 3)),
        #layers.MaxPooling3D(pool_size=(2, 2, 3)),
        layers.Dropout(dropoutrate),
        layers.Conv3D( filters = 40, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(2, 2, 1)),
        #layers.MaxPooling3D(pool_size=(2, 2, 1)),
        layers.Dropout(dropoutrate),
        layers.Conv3D( filters = 80, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(2, 2, 1)),
        #layers.MaxPooling3D(pool_size=(2, 2, 1)),
        layers.Conv3D( filters = 5, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropoutrate),
        layers.Conv3D( filters = 1, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Reshape((8,4,5)),
        layers.Conv2D( filters = 2, kernel_size = kernel, padding='same', activation='relu'),
        layers.Flatten(),
        #layers.Dense(dense, activation='relu'),
        layers.Dense(numClass, activation='softmax')
    ])
    return model
def build_ConvLSTM(kernel_size = 5, numClass = 9, dropoutrate = 0.5):
    model = keras.models.Sequential([
        tf.keras.layers.Input((128, 64, 50, 1)),
        tf.keras.layers.Permute((3,1,2,4)),
        layers.Bidirectional(layers.ConvLSTM2D( filters = 10, kernel_size = kernel_size, padding='same', activation='relu', return_sequences=True), merge_mode='sum'),
        layers.AveragePooling3D(pool_size=(3, 2, 2)),
        layers.Dropout(dropoutrate),
        layers.Bidirectional(layers.ConvLSTM2D( filters = 20, kernel_size = kernel_size, padding='same', activation='relu', return_sequences=True), merge_mode='sum'),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(3, 2, 2)),
        layers.Dropout(dropoutrate),
        layers.Bidirectional(layers.ConvLSTM2D( filters = 40, kernel_size = kernel_size, padding='same', activation='relu', return_sequences=True), merge_mode='sum'),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(1, 2, 2)),
        
        layers.Dropout(dropoutrate),
        layers.Bidirectional(layers.ConvLSTM2D( filters = 80, kernel_size = kernel_size, padding='same', activation='relu', return_sequences=True), merge_mode='sum'),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(1, 2, 2)),
        layers.Bidirectional(layers.ConvLSTM2D( filters = 5, kernel_size = kernel_size, padding='same', activation='relu', return_sequences=True), merge_mode='sum'),
        layers.BatchNormalization(),
        layers.Dropout(dropoutrate),
        layers.Bidirectional(layers.ConvLSTM2D( filters = 1, kernel_size = kernel_size, padding='same', activation='relu', return_sequences=False), merge_mode='sum'),
        layers.BatchNormalization(),
        # layers.AveragePooling2D(pool_size=(2,2)),
        layers.Conv2D( filters = 1, kernel_size = kernel_size, padding='same', activation='relu'),
        layers.Flatten(),
        # layers.Dense(128, activation='relu'),
        layers.Dense(numClass, activation='softmax')
    ])

    return model

def build_TConv_Imgnet(filters = 5, kernel = (2,2,5), fine_tune_at = 500, numClass = 9, imag_model = 'MobileNetV2', dropoutrate = 0.2):
    m_input = keras.Input(shape = (128, 64 ,50, 1))
    # TConv
    time_model = keras.Sequential([
        layers.Conv3D( filters = filters, kernel_size = kernel, padding='same', activation='relu', input_shape=(128, 64 ,50, 1)),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(1, 1, 4)),
        layers.Dropout(dropoutrate),
        layers.Conv3D( filters = filters, kernel_size = kernel, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.AveragePooling3D(pool_size=(1, 1, 4)),
        #layers.Dropout(dropoutrate),
        #layers.Conv3D( filters = filters*4, kernel_size = kernel, padding='same', activation='relu'),
        #layers.BatchNormalization(),
        #layers.AveragePooling3D(pool_size=(1, 1, 2)),
        #layers.Dropout(dropoutrate),
        #layers.Conv3D( filters = 5, kernel_size = kernel, padding='same', activation='relu'),
        #layers.BatchNormalization(),
        layers.Dropout(dropoutrate),
        layers.Conv3D( filters = 1, kernel_size = kernel, padding='same', activation='relu'),
        #layers.BatchNormalization(),
        #layers.AveragePooling3D(pool_size=(1, 1, 2)),
        #layers.Dropout(dropoutrate),
        layers.UpSampling3D(size=(1,2,1))
        ], name = 'ModelTimeConv')
    time_model.Name = 'ModelTimeConv'    
    # Inception
    if imag_model == 'InceptionResNetV2':
        imag_model = keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
    elif imag_model == 'EfficientNetB0':
        imag_model = tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(128,128,3))
    elif imag_model == 'MobileNetV2':
        imag_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
    for layer in imag_model.layers[:fine_tune_at]:
        layer.trainable = False
    imag_model.Name = 'ModelImageNet'
    # Post model
    post_model = keras.Sequential([
        layers.Conv2D(filters = 128, kernel_size = (1,1), padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu')
        ], name = 'ModelPost')
    post_model.Name = 'ModelPost'
    m_output = time_model(m_input)
    m_output = imag_model(m_output)
    m_output = post_model(m_output)
    #m_output = layers.Dropout(0.2)(m_output)
    m_output = layers.Dense(numClass, activation='softmax')(m_output)
    model = keras.Model(
        inputs = m_input,
        outputs = m_output,
    )
    return model

def build_Incpt_LSTM(fine_tune_at=700, numClass=9):
    m_input = keras.Input(shape = (128, 64 ,50, 1)) 
    # Inception
    imag_model = keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(128,128,3))
    for layer in imag_model.layers[:fine_tune_at]:
        layer.trainable = False
    imag_model.Name = 'ModelImageNet'
    # Post model
    post_model = keras.Sequential([
        layers.Conv2D(filters = 256, kernel_size = (1,1), padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu')
        ], name = 'ModelPost')
    post_model.Name = 'ModelPost'
    # LSTM model
    lstm_model = keras.models.Sequential([
        # Shape [batch, time, features] => [batch, 50, 5903]
        layers.LSTM(64, input_shape=(50, 256)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization()
    ])
    lstm_model.Name = 'ModelLSTM'
    return

class ImagRepeatingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ImagRepeatingLayer, self).__init__()
        #self.batches = batch_size
        #self.target_dim = (self.batches, 50, 128) # batch, seq, features
        #self.input_dim = (self.batches, 128, 64, 50) # batch, seq, features
        #self.out = tf.Variable(tf.ones(self.target_dim))
        self.fine_tune_at = 50
        #imag_model = keras.applications.InceptionResNetV2(weights='imagenet', 
        #                              include_top=False, input_shape=(128,128,3))
        imag_model = keras.applications.MobileNetV2(weights='imagenet', 
                                      include_top=False, input_shape=(128,128,3))
        for layer in imag_model.layers[:self.fine_tune_at]:
            layer.trainable = False
        imag_model.Name = 'ModelImageNet'
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 64)),
            tf.keras.layers.Reshape(target_shape=(128,64,1)),
            tf.keras.layers.UpSampling3D(size=(1,2,3)),  #128,128,3
            imag_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu')
            ])

    def call(self, x):
        output_list = []
        for t in range(x.shape[3]):#self.target_dim[1]):
            #print(t)
            #print(x[:,t,:].shape)
            #print(self.model(x[:,:,:,t]).shape)
            output_list.append(self.model(x[:,:,:,t]))
        self.out = tf.stack(output_list, axis=1)
        return self.out
    
def build_Img_TConv(numClass):
    filters = 5
    kernel=3
    time_model = keras.Sequential([
            layers.Reshape((50,128,1), input_shape=(50, 128)),
            layers.Conv2D( filters = filters, kernel_size = kernel, padding='same', activation='relu', input_shape=(50, 128, 1)),
            layers.BatchNormalization(),
            layers.AveragePooling2D(pool_size=(2,2)),
            layers.Dropout(0.2),
            layers.Conv2D( filters = filters, kernel_size = kernel, padding='same', activation='relu', input_shape=(25, 64, 1)),
            layers.BatchNormalization(),
            layers.AveragePooling2D(pool_size=(2,2)),
            layers.Dropout(0.2),
            layers.Conv2D( filters = filters, kernel_size = kernel, padding='same', activation='relu', input_shape=(12, 32, 1)),
            layers.BatchNormalization(),
            layers.AveragePooling2D(pool_size=(2,2)),
            layers.Dropout(0.2),
            layers.Conv2D( filters = filters, kernel_size = kernel, padding='same', activation='relu', input_shape=(6, 16, 1)),
            layers.BatchNormalization(),
            #layers.AveragePooling2D(pool_size=(2,1)),
            layers.Dropout(0.2),
            layers.Flatten()
            ], name = 'ModelTimeConv')
    time_model.Name = 'ModelTimeConv'   
        
    m_input = tf.keras.Input(shape = (128, 64 ,50, 1))
    m_output = ImagRepeatingLayer()(m_input)
    m_output = time_model(m_output)
    m_output = tf.keras.layers.Dense(numClass, activation='softmax')(m_output)
    model = keras.Model(
        inputs = m_input,
        outputs = m_output,
    )
    return model

class ImagTDRepeatingLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ImagTDRepeatingLayer, self).__init__()
        #self.batches = batch_size
        #self.target_dim = (self.batches, 50, 128) # batch, seq, features
        #self.input_dim = (self.batches, 128, 64, 50) # batch, seq, features
        #self.out = tf.Variable(tf.ones(self.target_dim))
        self.fine_tune_at = 30
        #imag_model = keras.applications.InceptionResNetV2(weights='imagenet', 
        #                              include_top=False, input_shape=(128,128,3))
        imag_model = keras.applications.MobileNetV2(weights='imagenet', 
                                      include_top=False, input_shape=(128,128,3))
        
        for layer in imag_model.layers[:self.fine_tune_at]:
            layer.trainable = False
        imag_model.Name = 'ModelImageNet'
        self.model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(128, 64)),
            #tf.keras.layers.Permute((2,3,1)),
            tf.keras.layers.Reshape(target_shape=(128,64,1,1)),
            tf.keras.layers.UpSampling3D(size=(1,2,3)),  #128,128,3
            imag_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu')
            ])

    def call(self, x):
        output_list = []
        for t in range( int(x.shape[1]/2)):#self.target_dim[1]):
            output_list.append(tf.keras.layers.TimeDistributed(self.model)(x[:,(t*2):(t+1)*2,:,:,:]))#(self.model(x[:,:,:,t]))
        self.out = tf.stack(output_list, axis=1)
        self.out = tf.keras.layers.Reshape((x.shape[1],128))(self.out)
        return self.out
    
def build_Img_TConv_TD(numClass):
    filters = 5
    kernel=3
    time_model = keras.Sequential([
            layers.Reshape((25,128,1), input_shape=(25, 128)),
            layers.Conv2D( filters = filters, kernel_size = kernel, padding='same', activation='relu', input_shape=(25, 128, 1)),
            layers.BatchNormalization(),
            layers.AveragePooling2D(pool_size=(2,2)),
            layers.Dropout(0.2),
            layers.Conv2D( filters = filters*2, kernel_size = kernel, padding='same', activation='relu', input_shape=(12, 64, 1)),
            layers.BatchNormalization(),
            layers.AveragePooling2D(pool_size=(2,2)),
            layers.Dropout(0.2),
            layers.Conv2D( filters = filters*4, kernel_size = kernel, padding='same', activation='relu', input_shape=(6, 32, 1)),
            #layers.BatchNormalization(),
            #layers.AveragePooling2D(pool_size=(2,2)),
            #layers.Dropout(0.2),
            #layers.Conv2D( filters = filters*8, kernel_size = kernel, padding='same', activation='relu', input_shape=(3, 16, 1)),
            layers.BatchNormalization(),
            layers.Conv2D( filters = 1, kernel_size = kernel, padding='same', activation='relu'),
            #layers.AveragePooling2D(pool_size=(2,1)),
            layers.Dropout(0.2),
            layers.Flatten()
            ], name = 'ModelTimeConv')
    time_model.Name = 'ModelTimeConv' 

    fine_tune_at = 30
    #imag_model = keras.applications.InceptionResNetV2(weights='imagenet', 
    #                              include_top=False, input_shape=(128,128,3))
    imag_model = tf.keras.applications.MobileNetV2(weights='imagenet', 
                                    include_top=False, input_shape=(128,128,3))
    
    def repeat(x):
        x1 = tf.keras.layers.concatenate( (x,x,x), axis=-1)
        return x1
    repeat_layer = tf.keras.layers.Lambda(repeat, output_shape=(128,64,3))

    imag_model_2 = tf.keras.Sequential([
        tf.keras.layers.Input((128, 64, 1)),
        #tf.keras.layers.Permute((2,3,1)),
        # tf.keras.layers.Reshape(target_shape=(128,64,1)),
        tf.keras.layers.UpSampling2D(size=(1,2),interpolation='bilinear'), # change to zero padding?
        repeat_layer,
        # tf.keras.layers.UpSampling3D(size=(1,1,3)),  #128,128,3 for RGB channels
        imag_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu')
        ])   
    
    m_input = tf.keras.Input((128, 64, 50, 1))
    m_output = tf.keras.layers.AveragePooling3D( pool_size=(1,1,2) )(m_input)
    m_output = tf.keras.layers.Permute((3,1,2,4))(m_output)

    m_output = tf.keras.layers.TimeDistributed(imag_model_2)(m_output)
    m_output = time_model(m_output)
    m_output = tf.keras.layers.Dense(numClass, activation='softmax')(m_output)
    model = keras.Model(
        inputs = m_input,
        outputs = m_output,
    )
    return model

def build_Img_LSTM_TD(numClass):
    lstm_model = keras.models.Sequential([
        # Shape [batch, time, features] => [batch, 50, 5903]
        layers.Bidirectional(layers.LSTM(64), merge_mode='concat'),
        # layers.LSTM(64),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization()
    ])
    fine_tune_at = 30
    #imag_model = keras.applications.InceptionResNetV2(weights='imagenet', 
    #                              include_top=False, input_shape=(128,128,3))
    imag_model = tf.keras.applications.MobileNetV2(weights='imagenet', 
                                    include_top=False, input_shape=(128,128,3))
    def repeat(x):
        x1 = tf.keras.layers.concatenate( (x,x,x), axis=-1)
        return x1
    repeat_layer = tf.keras.layers.Lambda(repeat, output_shape=(128,64,3))

    imag_model_2 = tf.keras.Sequential([
        tf.keras.layers.Input((128, 64, 1)),
        #tf.keras.layers.Permute((2,3,1)),
        # tf.keras.layers.Reshape(target_shape=(128,64,1)),
        tf.keras.layers.UpSampling2D(size=(1,2),interpolation='bilinear'), # change to zero padding?
        repeat_layer,
        # tf.keras.layers.UpSampling3D(size=(1,1,3)),  #128,128,3 for RGB channels
        imag_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu')
        ])   
    
    imag_model_2.summary()
    m_input = tf.keras.Input((128, 64, 50, 1))
    m_output = tf.keras.layers.AveragePooling3D( pool_size=(1,1,2) )(m_input)
    m_output = tf.keras.layers.Permute((3,1,2,4))(m_output)
    print('shape:',m_output.shape)
    m_output = tf.keras.layers.TimeDistributed(imag_model_2)(m_output)
    print(m_output.shape)
    m_output = lstm_model(m_output)
    m_output = tf.keras.layers.Dense(numClass, activation='softmax')(m_output)
    model = keras.Model(
        inputs = m_input,
        outputs = m_output,
    )
    return model


def build_Conv_Trans(num_heads = 8, dff = 64, numClass = 47, d_model = 64,
                     dropoutrate = 0.2, conv_filters = 10, conv_kernel = 5):
    model = trans_builder.build_Conv_Trans(num_heads = num_heads, dff = dff, 
                                               numClass = numClass, d_model = d_model,
                                               dropoutrate = dropoutrate, conv_filters = conv_filters, 
                                               conv_kernel = conv_kernel)
    return model

def build_NeoConv_Trans(num_heads = 8, dff = 64, numClass = 47, d_model = 64,
                     dropoutrate = 0.2, conv_filters = 10, conv_kernel = 5):
    model = trans_builder.build_NeoConv_Trans(num_heads = num_heads, dff = dff, 
                                               numClass = numClass, d_model = d_model,
                                               dropoutrate = dropoutrate, conv_filters = conv_filters, 
                                               conv_kernel = conv_kernel)
    return model

def build_Conv_Trans_w9(num_heads = 8, dff = 64, numClass = 47, d_model = 64,
                     dropoutrate = 0.2, conv_filters = 10, conv_kernel = 5, 
                     model_9_path = '../Outputs/TrainedModels/modelweight_Conv_Trans_9.h5'):
    model = trans_builder.build_Conv_Trans_w9(num_heads = num_heads, dff = dff, 
                                               numClass = numClass, d_model = d_model,
                                               dropoutrate = dropoutrate, conv_filters = conv_filters, 
                                               conv_kernel = conv_kernel, model_9_path = model_9_path)
    return model