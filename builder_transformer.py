# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:51:59 2021

@author: fredr
"""


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import math

def positional_encoding_3d(xlim = 64,ylim = 32,zlim = 25, d_model=32):
  scale=2 * math.pi
  temperature=10000
  volume = np.ones([1, xlim, ylim, zlim], dtype=float)
  x_embed = np.cumsum(volume, 1)
  y_embed = np.cumsum(volume, 2)
  z_embed = np.cumsum(volume, 3)

  eps = 1e-6
  x_embed = x_embed / (x_embed[:, -1:, :, :] + eps) * scale
  y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * scale
  z_embed = z_embed / (z_embed[:, :, :, -1:] + eps) * scale

  dim_t = np.arange(d_model, dtype=float)
  dim_t = temperature ** (2 * (dim_t // 2) / d_model)

  pos_x = x_embed[:, :, :, :, None] / dim_t
  pos_y = y_embed[:, :, :, :, None] / dim_t
  pos_z = z_embed[:, :, :, :, None] / dim_t

  pos_x[:, :, :, :, 0::2] = np.sin(pos_x[:, :, :, :, 0::2])
  pos_x[:, :, :, :, 1::2] = np.cos(pos_x[:, :, :, :, 1::2])
  pos_y[:, :, :, :, 0::2] = np.sin(pos_y[:, :, :, :, 0::2])
  pos_y[:, :, :, :, 1::2] = np.cos(pos_y[:, :, :, :, 1::2])
  pos_z[:, :, :, :, 0::2] = np.sin(pos_z[:, :, :, :, 0::2])
  pos_z[:, :, :, :, 1::2] = np.cos(pos_z[:, :, :, :, 1::2])

  pos = pos_x+pos_y+pos_z

  return tf.cast(pos, dtype=tf.float32), pos

def positional_encoding_4d(xlim = 64,ylim = 32,zlim = 25, clim = 5, d_model=32):
  scale=2 * math.pi
  temperature=10000
  volume = np.ones([1, xlim, ylim, zlim, clim], dtype=float)
  x_embed = np.cumsum(volume, 1)
  y_embed = np.cumsum(volume, 2)
  z_embed = np.cumsum(volume, 3)
  c_embed = np.cumsum(volume, 4)

  eps = 1e-6
  x_embed = x_embed / (x_embed[:, -1:, :, :, :] + eps) * scale
  y_embed = y_embed / (y_embed[:, :, -1:, :, :] + eps) * scale
  z_embed = z_embed / (z_embed[:, :, :, -1:, :] + eps) * scale
  c_embed = c_embed / (c_embed[:, :, :, :, -1:] + eps) * scale

  dim_t = np.arange(d_model, dtype=float)
  dim_t = temperature ** (2 * (dim_t // 2) / d_model)

  pos_x = x_embed[:, :, :, :, :, None] / dim_t
  pos_y = y_embed[:, :, :, :, :, None] / dim_t
  pos_z = z_embed[:, :, :, :, :, None] / dim_t
  pos_c = c_embed[:, :, :, :, :, None] / dim_t

  pos_x[:, :, :, :, :, 0::2] = np.sin(pos_x[:, :, :, :, :, 0::2])
  pos_x[:, :, :, :, :, 1::2] = np.cos(pos_x[:, :, :, :, :, 1::2])
  pos_y[:, :, :, :, :, 0::2] = np.sin(pos_y[:, :, :, :, :, 0::2])
  pos_y[:, :, :, :, :, 1::2] = np.cos(pos_y[:, :, :, :, :, 1::2])
  pos_z[:, :, :, :, :, 0::2] = np.sin(pos_z[:, :, :, :, :, 0::2])
  pos_z[:, :, :, :, :, 1::2] = np.cos(pos_z[:, :, :, :, :, 1::2])
  pos_c[:, :, :, :, :, 0::2] = np.sin(pos_c[:, :, :, :, :, 0::2])
  pos_c[:, :, :, :, :, 1::2] = np.cos(pos_c[:, :, :, :, :, 1::2])

  pos = pos_x + pos_y + pos_z + pos_c
  
  pos_flat = np.reshape(pos, (1, xlim*ylim*zlim*clim, d_model))
  
  return tf.cast(pos_flat, dtype=tf.float32)#, pos, pos_flat
#%%
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.dff = dff
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)#MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'd_model': self.d_model,
        'num_heads': self.num_heads,
        'dff': self.dff,
    })
    return config

  def call(self, x, training, mask):
    #print("x in encoder: "+str(x.shape))
    attn_output, _ = self.mha(x, x, return_attention_scores=True)#, x, return_attention_scores=True)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    #print("attn_output in encoder: "+str(attn_output.shape))
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, pos_dim = [8,4,5,5]):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.dff = dff
    self.pos_dim  = pos_dim
    self.pos_tf = positional_encoding_4d(pos_dim[0],pos_dim[1],pos_dim[2],pos_dim[3],d_model)
    self.exp_dim = tf.keras.layers.Reshape( (pos_dim[0]*pos_dim[1]*pos_dim[2]*pos_dim[3],1) )  #need to be same as positional encoding
    self.embedding = tf.keras.layers.Conv1D(filters = d_model, kernel_size=5, padding='same', activation='relu')
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)
  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'num_layers': self.num_layers,
        'd_model': self.d_model,
        'num_heads': self.num_heads,
        'dff': self.dff,
        'pos_dim': self.pos_dim,
    })
    return config
  def call(self, x, training, mask):
    # adding embedding and position encoding.    
    x = self.exp_dim(x)
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_tf[:,:,:]
    x = self.dropout(x, training=training)


    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)

def build_Conv_Trans(num_heads = 8, dff = 64, numClass = 47, d_model = 64,
                     dropoutrate = 0.2, conv_filters = 5, conv_kernel = 5):
    #m_input = tf.random.uniform((4, 32, 16, 8,1), dtype=tf.float32, minval=0, maxval=200)
    m_input = keras.Input(shape = (128, 64, 50, 1))
    m_output = layers.Conv3D( filters = conv_filters, kernel_size = conv_kernel, padding='same', activation='relu', input_shape=(128, 64 ,50, 1))(m_input)        
    m_output = layers.BatchNormalization()(m_output)
    m_output = layers.AveragePooling3D(pool_size=(2, 2, 3))(m_output)
    m_output = layers.Dropout(dropoutrate)(m_output)
    m_output = layers.Conv3D( filters = conv_filters, kernel_size = conv_kernel, padding='same', activation='relu')(m_output)        
    m_output = layers.BatchNormalization()(m_output)
    m_output = layers.AveragePooling3D(pool_size=(2, 2, 3))(m_output)
    m_output = layers.Dropout(dropoutrate)(m_output)
    m_output = layers.Conv3D( filters = conv_filters, kernel_size = conv_kernel, padding='same', activation='relu')(m_output)        
    m_output = layers.BatchNormalization()(m_output)
    m_output = layers.AveragePooling3D(pool_size=(2, 2, 1))(m_output)
    m_output = layers.Dropout(dropoutrate)(m_output)
    m_output = layers.Conv3D( filters = conv_filters, kernel_size = conv_kernel, padding='same', activation='relu')(m_output)        
    m_output = layers.BatchNormalization()(m_output)
    m_output = layers.AveragePooling3D(pool_size=(2, 2, 1))(m_output)
    m_output = layers.Dropout(dropoutrate)(m_output)
    print(m_output.shape[1:].as_list())       
    pos_dim = m_output.shape[1:].as_list()
    #m_output = m_input# + pos_tf
    #print("input: "+str(m_output.shape))
    m_output = layers.Flatten()(m_output)
    print("m_output before encoder: "+str(m_output.shape))
    #m_output = EncoderLayer(d_model = d_model, num_heads=num_heads, dff=dff)(m_output, False, None)
    m_output = Encoder(num_layers = 5, d_model = d_model, num_heads=num_heads, dff=dff, pos_dim=pos_dim)(m_output, False, None)

    #print("m_output after encoder: "+str(m_output.shape))
    #m_output = layers.AveragePooling1D( pool_size = 5, strides= 5 )(m_output)
    m_output = layers.LocallyConnected1D(filters = 8, kernel_size = 5, strides= 5)(m_output)
    #m_output = layers.Permute(dims = (2,1))(m_output)
    m_output = layers.Flatten()(m_output)
    #print("m_output after flatten: "+str(m_output.shape))
    m_output = layers.Dropout(dropoutrate)(m_output)
    m_output = layers.BatchNormalization()(m_output)
    m_output = layers.Dense(64, activation='relu')(m_output)
    #m_output = layers.Dropout(0.2)(m_output)
    m_output = layers.Dense(numClass, activation = 'softmax')(m_output)

    model = keras.Model(
    inputs = m_input,
    outputs = m_output,
    )
    return model

def build_Conv_Trans_w9(num_heads = 8, dff = 64, numClass = 47, d_model = 64,
                     dropoutrate = 0.2, conv_filters = 5, conv_kernel = 5, 
                     model_9_path = '../Outputs/TrainedModels/modelweight_Conv_Trans_9.h5'):
    #m_input = tf.random.uniform((4, 32, 16, 8,1), dtype=tf.float32, minval=0, maxval=200)
    m_input = keras.Input(shape = (128, 64, 50, 1))
    m_output = layers.Conv3D( filters = conv_filters, kernel_size = conv_kernel, padding='same', activation='relu', input_shape=(128, 64 ,50, 1))(m_input)        
    m_output = layers.BatchNormalization()(m_output)
    m_output = layers.AveragePooling3D(pool_size=(2, 2, 3))(m_output)
    m_output = layers.Dropout(dropoutrate)(m_output)
    m_output = layers.Conv3D( filters = conv_filters, kernel_size = conv_kernel, padding='same', activation='relu')(m_output)        
    m_output = layers.BatchNormalization()(m_output)
    m_output = layers.AveragePooling3D(pool_size=(2, 2, 3))(m_output)
    m_output = layers.Dropout(dropoutrate)(m_output)
    m_output = layers.Conv3D( filters = conv_filters, kernel_size = conv_kernel, padding='same', activation='relu')(m_output)        
    m_output = layers.BatchNormalization()(m_output)
    m_output = layers.AveragePooling3D(pool_size=(2, 2, 1))(m_output)
    m_output = layers.Dropout(dropoutrate)(m_output)
    m_output = layers.Conv3D( filters = conv_filters, kernel_size = conv_kernel, padding='same', activation='relu')(m_output)        
    m_output = layers.BatchNormalization()(m_output)
    m_output = layers.AveragePooling3D(pool_size=(2, 2, 1))(m_output)
    m_output = layers.Dropout(dropoutrate)(m_output)
    print(m_output.shape[1:].as_list())       
    pos_dim = m_output.shape[1:].as_list()
    #m_output = m_input# + pos_tf
    #print("input: "+str(m_output.shape))
    m_output = layers.Flatten()(m_output)
    print("m_output before encoder: "+str(m_output.shape))
    #m_output = EncoderLayer(d_model = d_model, num_heads=num_heads, dff=dff)(m_output, False, None)
    m_output = Encoder(num_layers = 5, d_model = d_model, num_heads=num_heads, dff=dff, pos_dim=pos_dim)(m_output, False, None)

    #print("m_output after encoder: "+str(m_output.shape))
    #m_output = layers.AveragePooling1D( pool_size = 5, strides= 5 )(m_output)
    m_output = layers.LocallyConnected1D(filters = 8, kernel_size = 5, strides= 5)(m_output)
    #m_output = layers.Permute(dims = (2,1))(m_output)
    m_output = layers.Flatten()(m_output)
    #print("m_output after flatten: "+str(m_output.shape))
    m_output = layers.Dropout(dropoutrate)(m_output)
    m_output = layers.BatchNormalization()(m_output)
    m_output = layers.Dense(64, activation='relu')(m_output)
    
    model_9 = build_Conv_Trans(num_heads = 8, dff = 32, numClass = 9, d_model = 32,
                     dropoutrate = 0.2, conv_filters = 5, conv_kernel = 3)
    model_9.load_weights(model_9_path)
    model_9.trainable=False
    m_output = layers.Concatenate()([m_output, model_9(m_input)])
    
    #m_output = layers.Dropout(0.2)(m_output)
    m_output = layers.Dense(numClass, activation = 'softmax')(m_output)

    model = keras.Model(
    inputs = m_input,
    outputs = m_output,
    )
    return model