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
    def __init__(self, data_format = 'channels_first', x_output_resolution = 256, dx_shape = [128, 32], post_conv1d_block_number = 2, post_dx_conv_block_number = 5, initial_kernel_size = 19, final_kernel_size = 3, use_batchnorm = False, kernel_regularizer = None, bias_regularizer = None, conv1d_layer_number = 3, conv1d_final_channels = 256, **kwargs):
        super().__init__(**kwargs)
        self.x_output_resolution = x_output_resolution
        self.data_format = data_format
        self.dx_upsample_channels = 16
        self.post_dx_conv_block_number = post_dx_conv_block_number
        self.dx_shape = dx_shape
        self.use_batchnorm = use_batchnorm
        self.training = True
        
        self.conv1d_layers = []
        for k in range(conv1d_layer_number):
            channels = conv1d_final_channels//(conv1d_layer_number - k)
            self.conv1d_layers.append(tf.keras.layers.Conv1D(filters = channels, kernel_size = initial_kernel_size, padding = 'same', activation = tf.nn.leaky_relu, data_format = self.data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))

        self.post_conv1d_blocks = []
        for k in range(post_conv1d_block_number):
            filters = self.dx_upsample_channels//(post_conv1d_block_number - k)
            block = []
            block.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = initial_kernel_size, padding = 'same', activation = tf.nn.leaky_relu, data_format = data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            block.append(ResnetBlock(data_format = data_format, filters = filters, kernel_size = initial_kernel_size, activation = tf.nn.leaky_relu, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            self.post_conv1d_blocks.append(block)
        # self.conv1d_0 = tf.keras.layers.Conv1D(filters=8, kernel_size = initial_kernel_size, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        # self.conv1d_1 = tf.keras.layers.Conv1D(filters=64, kernel_size = initial_kernel_size, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        # self.conv1d_2 = tf.keras.layers.Conv1D(filters=256, kernel_size = initial_kernel_size, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        # self.resnetblocks_0 = [ResnetBlock(data_format = data_format, filters = self.dx_upsample_channels, kernel_size = initial_kernel_size, activation = tf.nn.leaky_relu) for i in range(2)]
                                                  
        self.dx_dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dx_dense_1 = tf.keras.layers.Dense(256, activation = tf.nn.leaky_relu)
        self.dx_dense_2 = tf.keras.layers.Dense(tf.reduce_prod(dx_shape), activation = tf.nn.leaky_relu)
        #self.dx_resampled_conv_blocks = [ResampledConvolutionBlock([0.5**i, 0.5**i], data_format = data_format, filters = self.dx_upsample_channels, kernel_size = int(tf.reduce_min([initial_kernel_size] + [dim*0.5**i for dim in dx_shape])), activation = tf.nn.leaky_relu, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer) for i in range(4)]
        self.dx_resampled_conv_blocks = [AveragePoolingBlock(pool_size = 2**i, data_format = data_format, filters = self.dx_upsample_channels, use_resnetblocks = True, use_deconv_upsample = True, activation = tf.nn.leaky_relu, kernel_size = 10-2*i) for i in range(4)]
        self.dx_upsample = Upsample([-1, -1], data_format = data_format)

        self.post_dx_conv_blocks = []
        for k in range(self.post_dx_conv_block_number):
            ksize = initial_kernel_size + (k*(final_kernel_size - initial_kernel_size)//(post_dx_conv_block_number-1))
            filters = self.dx_upsample_channels-((k*self.dx_upsample_channels)//(post_dx_conv_block_number))
            block = []
            block.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = ksize, activation = tf.nn.leaky_relu, data_format = self.data_format, padding = 'same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            block.append(ResnetBlock(filters = filters, kernel_size = ksize , activation = tf.nn.leaky_relu, data_format = self.data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            self.post_dx_conv_blocks.append(block)

        if use_batchnorm:
            self.batchnorm_layers = []
            if self.data_format == 'channels_first':
                self.batchnorm_last = tf.keras.layers.BatchNormalization(axis = 1)
            else:
                self.batchnorm_last = tf.keras.layers.BatchNormalization(axis = -1)
            for k in range(post_dx_conv_block_number):
                if self.data_format == 'channels_first':
                    self.batchnorm_layers.append(tf.keras.layers.BatchNormalization(axis = 1))
                else:
                    self.batchnorm_layers.append(tf.keras.layers.BatchNormalization(axis = -1))
        
        self.conv_last = tf.keras.layers.Conv2D(filters = 1, kernel_size = final_kernel_size, activation = 'linear', padding = 'same', data_format = data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        self.resnet_last = ResnetBlock(filters = 1, kernel_size = final_kernel_size, activation = 'linear', data_format = data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)

    def call(self, inp):
       self.dx = inp[1] 
       try:
           if inp[2].shape[0] == 1:
               self.x_output_resolution = int(inp[2])
           elif self.data_format == 'channels_first':
               self.x_output_resolution = int(inp[2][2])
           else:
               self.x_output_resolution = int(inp[2][1])
       except:
           pass

       out = self.conv1d_layers[0](inp[0])
       for layer in self.conv1d_layers[1:]:
           out = layer(out)

       if self.data_format == 'channels_first':
           out = tf.expand_dims(out, axis = 1)
           domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx())], axis = 0))
           input_length = inp[0].shape[2]
       else:
           out = tf.expand_dims(out, axis = -1)
           domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[1]), tf.keras.backend.floatx())], axis = 0))
           input_length = inp[0].shape[1]

       for block in self.post_conv1d_blocks:
           for layer in block:
               out = layer(out)

       dx_res = self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(domain_info)))
       if self.data_format == 'channels_first':
           dx_res = tf.keras.backend.reshape(dx_res, (-1, 1, self.dx_shape[0], self.dx_shape[1]))
       else:
           dx_res = tf.keras.backend.reshape(dx_res, (-1, self.dx_shape[0], self.dx_shape[1], 1))
       dx_upsample_result = self.dx_resampled_conv_blocks[0](dx_res)
       for layer in self.dx_resampled_conv_blocks[1:]:
           dx_upsample_result += layer(dx_res)

       try:
           out = self.dx_upsample([out, [self.x_output_resolution, input_length]]) + self.dx_upsample([dx_upsample_result, [self.x_output_resolution, input_length]])
       except:
           out = self.dx_upsample([out, [64, input_length]]) + self.dx_upsample([dx_upsample_result, [64, input_length]])

       for block_num, block in enumerate(self.post_dx_conv_blocks):
           for layer in block:
               out = layer(out)
           if self.use_batchnorm:
               out = self.batchnorm_layers[block_num](out, training = self.training)

       if self.use_batchnorm:
           out = self.batchnorm_last(out, training = self.training)

       out = self.conv_last(out)
       out = self.resnet_last(out)
       return out
                
                
            
    
    # def call(self, inp):
    #     self.dx = inp[1]
    #     out = self.conv1d_2(self.conv1d_1(self.conv1d_0(inputs[0])))
    #     try:
    #         if inputs[2].shape[0] == 1:
    #             self.x_output_resolution = int(inputs[2])
    #         elif self.data_format == 'channels_first':
    #             self.x_output_resolution = int(inputs[2][2])
    #         else:
    #             self.x_output_resolution = int(inputs[2][1])
    #     except:
    #        pass

    #     try: #stupid hack 1 to get past keras '?' tensor dimensions via try except block
    #         if self.data_format == 'channels_first':
    #             self.domain_info = oe.contract('ij,j->ij',tf.tile(inputs[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[2]), tf.keras.backend.floatx())], axis = 0), backend = 'tensorflow')
    #             self.input_length = inputs[0].shape[-1]
    #             out = tf.expand_dims(out, axis = 1)
    #             dx_res = tf.reshape(self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info))), (-1, 1, 192, 64))
    #         else:
    #             self.domain_info = oe.contract('ij,j->ij',tf.tile(inputs[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[1]), tf.keras.backend.floatx())], axis = 0), backend = 'tensorflow')
    #             self.input_length = inputs[0].shape[-2]
    #             out = tf.expand_dims(out, axis = 3)
    #             dx_res = tf.reshape(self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info))), (-1, 192, 64, 1))
    #     except:
    #         if self.data_format == 'channels_first':
    #             self.domain_info = tf.tile(inputs[1], [1,3])
    #             self.input_length = inputs[0].shape[-1]
    #             out = tf.expand_dims(out, axis = 1)
    #             dx_res = tf.reshape(self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info))), (-1, 1, 192, 64))
    #         else:
    #             self.domain_info = tf.tile(inputs[1], [1,3])
    #             self.input_length = inputs[0].shape[-2]
    #             out = tf.expand_dims(out, axis = 3)
    #             dx_res = tf.reshape(self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info))), (-1, 192, 64, 1))
        
    #     for rb in self.resnetblocks_0:
    #         out = rb(out)
    #     try:
    #         out = self.dx_upsample([sum([rcb(dx_res) for rcb in self.dx_resampled_conv_blocks]), [self.x_output_resolution, self.input_length]]) + self.dx_upsample([out, [self.x_output_resolution, self.input_length]])
    #     except:
    #         out = self.dx_upsample([sum([rcb(dx_res) for rcb in self.dx_resampled_conv_blocks]), [64, self.input_length]]) + self.dx_upsample([out, [64, self.input_length]])
    #     for rb in self.resnetblocks_1:
    #         out = rb(out)

    #     return self.conv2d_2(self.bnorm_1(self.conv2d_1(self.bnorm_0(self.conv2d_0(out)))))

            
        
        
