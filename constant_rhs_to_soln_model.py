import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Conv2DTranspose, AveragePooling2D, Add, Reshape
from MergeWithAttention import MergeWithAttention2
from Upsample import Upsample
from collections.abc import Iterable

def constant_rhs_to_soln_dense(output_shape, load_weights_from = None):
    
    input_0 = Input(shape = (1,))
    
    if isinstance(output_shape, Iterable) and len(output_shape) == 2:
        dense_0 = Dense(output_shape[0]*output_shape[1])(input_0)
        reshape_0 = Reshape((1,output_shape[0],output_shape[1]))(dense_0)
    elif isinstance(output_shape, int):
        dense_0 = Dense(output_shape**2)(input_0)
        reshape_0 = Reshape((1,output_shape, output_shape))(dense_0)
    else:
        raise(TypeError('output_shape must be an integer or a tuple of 2 integers'))
        
    conv_0 = Conv2D(filters = 16, kernel_size = 3, activation=tf.nn.leaky_relu, data_format='channels_first', padding='same')(reshape_0)
    
    conv_1 = Conv2D(filters = 8, kernel_size = 3, activation=tf.nn.leaky_relu, data_format='channels_first', padding='same')(conv_0)
    
    conv_2 = Conv2D(filters = 1, kernel_size = 3, activation=tf.nn.tanh, data_format='channels_first', padding='same')(conv_1)
    
    mod = Model(input_0, conv_2)
    
    if load_weights_from:
        mod.load_weights(load_weights_from)
        
    return mod
        