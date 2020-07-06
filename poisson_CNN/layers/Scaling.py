import tensorflow as tf
import numpy as np

from .SpatialPyramidPool import SpatialPyramidPool
from ..utils import choose_conv_layer, get_pooling_method
'''
class Scaling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scaling_weight = self.add_weight(name = 'scaling_weight', shape = (), initializer=tf.initializers.ones, trainable = True)

    def call(self, inp):
        return inp * self.scaling_weight
'''

class Scaling(tf.keras.layers.Layer):
    def __init__(self, ndims, stages = 2, downsampling_ratio_per_stage = 2, data_format = 'channels_first', padding = 'same', spp_levels = [[2,2], 3, 5], **convargs):
        super().__init__()
        self.ndims = ndims
        self.spp = SpatialPyramidPool(ndims = self.ndims, levels = spp_levels, data_format = data_format, pooling_type = 'MAX')
        self.stages = []
        self.data_format = data_format
        pool_layer = get_pooling_method('Average', self.ndims)
        conv_layer = choose_conv_layer(self.ndims)
        for k in range(stages):
            self.stages.append(conv_layer(data_format = data_format, padding = padding, **convargs))
            self.stages.append(pool_layer(pool_size = downsampling_ratio_per_stage, padding = padding, data_format = data_format))

        self.dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dense_1 = tf.keras.layers.Dense(25, activation = tf.nn.leaky_relu)
        self.dense_2 = tf.keras.layers.Dense(1, activation = 'linear')

    @tf.function
    def call(self, inp):
        #input format: a list of 2 tf.Tensors such that [input_to_scale, 2nd input] where the 1st tensor has a shape compatible with tf.keras.layers.Conv2D/Conv3D
        # if self.data_format == 'channels_first' and isinstance(inp, list):
        #     inp = tf.concat(inp, axis = 1)
        #     inp_to_scale = tf.expand_dims(inp[:,0,...], axis = 1)
        # elif self.data_format == 'channels_first':
        #     inp_to_scale = tf.expand_dims(inp[:,0,...], axis = 1)
        # elif isinstance(inp,list):
        #     inp = tf.concat(inp, axis = -1)
        #     inp_to_scale = tf.expand_dims(inp[:,...,0], axis = -1)
        # else:
        #     inp_to_scale = tf.expand_dims(inp[:,...,0], axis = -1)
        inp_to_scale = inp[0]
        inp = tf.concat(inp, axis = 1 if self.data_format == 'channels_first' else -1)
        out = self.stages[0](inp)
        for k in range(1, len(self.stages)):
            out = self.stages[k](out)
        out = self.spp(out)
        out = self.dense_2(self.dense_1(self.dense_0(out)))
        return tf.einsum('i...,i...->i...', 1.0+out, inp_to_scale)

if __name__ == '__main__':
    conv_args = {'filters': 2, 'kernel_size': 3}
    mod = Scaling(2, **conv_args)
    inp = [tf.random.uniform((10,2,100,100)), tf.random.uniform((10,1,100,100))]
    print(mod(inp).shape)
