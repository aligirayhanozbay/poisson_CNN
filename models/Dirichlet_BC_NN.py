import tensorflow as tf
import itertools
import opt_einsum as oe
import numpy as np
from abc import ABC, abstractmethod
import sys
sys.path.append('..')

from ..layers import WeightedContractionLayer
from ..layers import Upsample
from .Model_With_Integral_Loss import Model_With_Integral_Loss_ABC
from .custom_blocks import *

#best candidate
class Dirichlet_BC_NN_2B(Model_With_Integral_Loss_ABC): #variant to include dx info
    def __init__(self, data_format = 'channels_first', x_output_resolution = 256, n_sepconvblocks = 3, **kwargs):
        super().__init__(**kwargs)
        self.x_output_resolution = x_output_resolution
        self.data_format = data_format
        self.conv1d_0 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        if data_format == 'channels_first':
            sepconvkernelshape = (256,5)
        else:
            sepconvkernelshape = (5,256)
        self.sepconvblocks_3 = [SepConvBlock(data_format = data_format, separable_kernel_size = sepconvkernelshape, separable_filters = 10, nonsep_filters = 10) for i in range(n_sepconvblocks)]
        
        self.conv2d_4 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        
        self.output_upsample_4 = Upsample([-1, -1], data_format = data_format , resize_method = tf.image.ResizeMethod.BICUBIC)
        
        self.conv2d_5 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 9, activation = tf.tanh, padding = 'same', data_format = data_format)
        
        self.dx_dense_0 = tf.keras.layers.Dense(4, activation = tf.nn.relu)
        self.dx_dense_1 = tf.keras.layers.Dense(4, activation = tf.nn.relu)
        self.dx_dense_2 = tf.keras.layers.Dense(8, activation = tf.nn.softmax)
    def call(self, inputs):
        self.dx = inputs[1]
        dx_res = 1/(1e-8 + 10 * self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(inputs[1]))))
        
        if self.data_format == 'channels_first':
            self.input_length = inputs[0].shape[-1]
            contr_expr = 'ijk,ij->ijk'
        else:
            self.input_length = inputs[0].shape[-2]
            contr_expr = 'ikj,ij->ikj'
        out = self.conv1d_2(tf.einsum(contr_expr, self.conv1d_1(self.conv1d_0(inputs[0])), dx_res))
        
        if self.data_format == 'channels_first':
            out = tf.expand_dims(out, axis = 1)
            newshape = [self.x_output_resolution, self.input_length]
        else:
            out = tf.expand_dims(out, axis = 3)
            newshape = [self.input_length, self.x_output_resolution]
            
        for scb in self.sepconvblocks_3:
            out = scb(out)
            
        out = self.output_upsample_4([self.conv2d_4(out), newshape])
        return self.conv2d_5(out)
    
class Dirichlet_BC_NN_2C(Model_With_Integral_Loss_ABC): #variant to include dx AND Lx/Ly info
    def __init__(self, data_format = 'channels_first', x_output_resolution = 256, n_sepconvblocks = 3, **kwargs):
        super().__init__(**kwargs)
        self.x_output_resolution = x_output_resolution
        self.data_format = data_format
        self.conv1d_0 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        if data_format == 'channels_first':
            sepconvkernelshape = (256,5)
        else:
            sepconvkernelshape = (5,256)
        self.sepconvblocks_3 = [SepConvBlock(data_format = data_format, separable_kernel_size = sepconvkernelshape, separable_filters = 10, nonsep_filters = 10) for i in range(n_sepconvblocks+1)]
        
        self.conv2d_4_0 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        self.conv2d_4_1 = tf.keras.layers.Conv2D(filters = 12, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        self.conv2d_4_2 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        
        
        self.output_upsample_4 = Upsample([-1, -1], data_format = data_format , resize_method = tf.image.ResizeMethod.BICUBIC)
        
        self.conv2d_5_0 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 9, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        
        self.conv2d_5_1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 9, activation = tf.nn.tanh, padding = 'same', data_format = data_format)
        
        
        self.dx_dense_0 = tf.keras.layers.Dense(8, activation = tf.nn.leaky_relu)
        self.dx_dense_1 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dx_dense_2 = tf.keras.layers.Dense(256, activation = tf.nn.tanh)
    def call(self, inputs):
        self.x_output_resolution = inputs[2]
        
        self.dx = inputs[1]
        try: #stupid hack 1 to get past keras '?' tensor dimensions via try except block
            if self.data_format == 'channels_first':
                self.domain_info = oe.contract('ij,j->ij',tf.tile(inputs[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[2]), tf.keras.backend.floatx())], axis = 0), backend = 'tensorflow')
                self.input_length = inputs[0].shape[-1]
                contr_expr = 'ikl,ik->ikl'
            else:
                self.domain_info = oe.contract('ij,j->ij',tf.tile(inputs[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[1]), tf.keras.backend.floatx())], axis = 0), backend = 'tensorflow')
                self.input_length = inputs[0].shape[-2]
                contr_expr = 'ilk,ik->ilk'
        except:
            if self.data_format == 'channels_first':
                self.domain_info = tf.tile(inputs[1], [1,3])
                self.input_length = inputs[0].shape[-1]
                contr_expr = 'ikl,ik->ikl'
            else:
                self.domain_info = tf.tile(inputs[1], [1,3])
                self.input_length = inputs[0].shape[-2]
                contr_expr = 'ilk,ik->ilk'
        dx_res = self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info)))

        out = tf.einsum(contr_expr, self.conv1d_2(self.conv1d_1(self.conv1d_0(inputs[0]))), dx_res)
        try: #stupid hack 2 to get past keras '?' tensor dims via try except block
            if self.data_format == 'channels_first':
                out = tf.expand_dims(out, axis = 1)
                newshape = [int(self.x_output_resolution), self.input_length]
            else:
                out = tf.expand_dims(out, axis = 3)
                newshape = [self.input_length, int(self.x_output_resolution)]
            
            for scb in self.sepconvblocks_3:
                out = scb(out)

            out = self.output_upsample_4([self.conv2d_4_2(self.conv2d_4_1(self.conv2d_4_0(out))), newshape])
            return self.conv2d_5_1(self.conv2d_5_0(out))
        except:
            if self.data_format == 'channels_first': # 
                #out = tf.expand_dims(out, axis = 1)
                newshape = [64, self.input_length]
            else:
                #out = tf.expand_dims(out, axis = 3)
                newshape = [self.input_length, 64]
            for scb in self.sepconvblocks_3:
                out = scb(out)
            
            out = self.output_upsample_4([self.conv2d_4_2(self.conv2d_4_1(self.conv2d_4_0(out))), newshape])
            return self.conv2d_5_1(self.conv2d_5_0(out))
        
                                                  
class Dirichlet_BC_NN_2D(Model_With_Integral_Loss_ABC):
    def __init__(self, data_format = 'channels_first', x_output_resolution = 256, **kwargs):
        super().__init__(**kwargs)
        self.x_output_resolution = x_output_resolution
        self.data_format = data_format
        
        self.conv1d_0 = tf.keras.layers.Conv1D(filters=8, kernel_size=13, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=13, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=13, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        self.resnetblocks_0 = [ResnetBlock(data_format = data_format, filters = 16, kernel_size = 9, activation = tf.nn.leaky_relu) for i in range(8)]
                                                  
        self.dx_dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dx_dense_1 = tf.keras.layers.Dense(256, activation = tf.nn.leaky_relu)
        self.dx_dense_2 = tf.keras.layers.Dense(64*192, activation = tf.nn.leaky_relu)
        self.dx_resampled_conv_blocks = [ResampledConvolutionBlock([0.5**i, 0.5**i], data_format = data_format, filters = 16, kernel_size = 19, activation = tf.nn.leaky_relu) for i in range(4)]
        self.dx_upsample = Upsample([-1, -1], data_format = data_format)

        self.resnetblocks_1 = [ResnetBlock(data_format = data_format, filters = 16, kernel_size = 9, activation = tf.nn.leaky_relu) for i in range(3)]
        
        self.conv2d_0 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 21, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        self.bnorm_0 = tf.keras.layers.BatchNormalization(axis = 1)
        self.conv2d_1 = tf.keras.layers.Conv2D(filters = 4, kernel_size = 21, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        self.bnorm_1 = tf.keras.layers.BatchNormalization(axis = 1)
        self.conv2d_2 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 21, activation = tf.nn.tanh, padding = 'same', data_format = data_format)
                                            
    def call(self, inputs):
        self.dx = inputs[1]
        out = self.conv1d_2(self.conv1d_1(self.conv1d_0(inputs[0])))
        try:
            if inputs[2].shape[0] == 1:
                self.x_output_resolution = int(inputs[2])
            elif self.data_format == 'channels_first':
                self.x_output_resolution = int(inputs[2][2])
            else:
                self.x_output_resolution = int(inputs[2][1])
        except:
           pass

        try: #stupid hack 1 to get past keras '?' tensor dimensions via try except block
            if self.data_format == 'channels_first':
                self.domain_info = oe.contract('ij,j->ij',tf.tile(inputs[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[2]), tf.keras.backend.floatx())], axis = 0), backend = 'tensorflow')
                self.input_length = inputs[0].shape[-1]
                out = tf.expand_dims(out, axis = 1)
                dx_res = tf.reshape(self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info))), (-1, 1, 192, 64))
            else:
                self.domain_info = oe.contract('ij,j->ij',tf.tile(inputs[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[1]), tf.keras.backend.floatx())], axis = 0), backend = 'tensorflow')
                self.input_length = inputs[0].shape[-2]
                out = tf.expand_dims(out, axis = 3)
                dx_res = tf.reshape(self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info))), (-1, 192, 64, 1))
        except:
            if self.data_format == 'channels_first':
                self.domain_info = tf.tile(inputs[1], [1,3])
                self.input_length = inputs[0].shape[-1]
                out = tf.expand_dims(out, axis = 1)
                dx_res = tf.reshape(self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info))), (-1, 1, 192, 64))
            else:
                self.domain_info = tf.tile(inputs[1], [1,3])
                self.input_length = inputs[0].shape[-2]
                out = tf.expand_dims(out, axis = 3)
                dx_res = tf.reshape(self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info))), (-1, 192, 64, 1))
        
        for rb in self.resnetblocks_0:
            out = rb(out)
        try:
            out = self.dx_upsample([sum([rcb(dx_res) for rcb in self.dx_resampled_conv_blocks]), [self.x_output_resolution, self.input_length]]) + self.dx_upsample([out, [self.x_output_resolution, self.input_length]])
        except:
            out = self.dx_upsample([sum([rcb(dx_res) for rcb in self.dx_resampled_conv_blocks]), [64, self.input_length]]) + self.dx_upsample([out, [64, self.input_length]])
        for rb in self.resnetblocks_1:
            out = rb(out)

        return self.conv2d_2(self.bnorm_1(self.conv2d_1(self.bnorm_0(self.conv2d_0(out)))))

            
        
        
