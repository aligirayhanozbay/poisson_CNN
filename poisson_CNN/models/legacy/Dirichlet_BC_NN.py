import tensorflow as tf
import itertools
import opt_einsum as oe
import numpy as np
from abc import ABC, abstractmethod
import sys
sys.path.append('..')

from ...layers import WeightedContractionLayer
from ...layers import Upsample
from .Model_With_Integral_Loss import Model_With_Integral_Loss_ABC
from .custom_blocks import *
                 
class Dirichlet_BC_NN(Model_With_Integral_Loss_ABC):
    '''
    This model takes a Dirichlet boundary condition imposed on a flat domain, the grid spacing and the desired output shape and outputs the solution to the Laplace equation with 1 Dirichlet BC (rest of the BCs are homogeneous)
    '''
    def __init__(self, data_format = 'channels_first', x_output_resolution = 256, dx_shape = [128, 32], post_conv1d_block_number = 2, post_dx_conv_block_number = 5, initial_kernel_size = 19, final_kernel_size = 3, use_batchnorm = False, kernel_regularizer = None, bias_regularizer = None, conv1d_layer_number = 3, conv1d_final_channels = 256, **kwargs):
        '''
        Init arguments:

        data_format: same as keras
        x_output_resolution: integer. no of gridpoints of the output in the direction perpendicular to the boundary. can be changed at runtime.
        dx_shape: list of integers. the final dense layer processing domain will have a no of units corresponding to the product of the members of this list. the output is then reshaped to match this.
        conv1d_layer_number: integer. no of 1d convolution layers at the start.
        conv1d_final_channels: integer. no of channels the final conv1d layer has.
        post_conv1d_block_number: integer. no of convolution blocks after the result of the initial 1d convs is reshaped to 2d.
        post_dx_conv_block_number: integer. no of convolution blocks after the result of the domain info processing is merged.
        initial_kernel_size: integer. initial kernel size of convolutions. most convolutions will have this kernel size.
        final_kernel_size: integer. the kernel size of the final convolutions will progressively move towards this number. keep it small to prevent output artefacting near image edges.
        use_batchnorm: boolean. determines if batch norm layers should be used. if set to true, supply __call__ argument training = False when performing inference.
        kernel_regularizer: same as the corresponding tf.keras.layers.Conv2D argument
        bias_regularizer: same as the corresponding tf.keras.layers.Conv2D argument

        **kwargs: used to set self.integral_loss parameters for the Model_With_Integral_Loss_ABC abstract base class
        '''
        super().__init__(**kwargs)
        self.x_output_resolution = x_output_resolution
        self.data_format = data_format
        self.dx_upsample_channels = 16
        self.post_dx_conv_block_number = post_dx_conv_block_number #no of conv blocks after merging doman information
        self.dx_shape = dx_shape
        self.use_batchnorm = use_batchnorm
        self.training = True
        
        self.conv1d_layers = []
        for k in range(conv1d_layer_number): #1d convs to convert 1d BC info to 2d data
            channels = conv1d_final_channels//(conv1d_layer_number - k)
            self.conv1d_layers.append(tf.keras.layers.Conv1D(filters = channels, kernel_size = initial_kernel_size, padding = 'same', activation = tf.nn.leaky_relu, data_format = self.data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))

        self.post_conv1d_blocks = [] #2d convs to increase channel size of the reshaped 1d conv results
        for k in range(post_conv1d_block_number):
            filters = self.dx_upsample_channels//(post_conv1d_block_number - k)
            block = []
            block.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = initial_kernel_size, padding = 'same', activation = tf.nn.leaky_relu, data_format = data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            block.append(ResnetBlock(data_format = data_format, filters = filters, kernel_size = initial_kernel_size, activation = tf.nn.leaky_relu, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            self.post_conv1d_blocks.append(block)
                                                  
        self.dx_dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu) #dense layers that handle domain shape info
        self.dx_dense_1 = tf.keras.layers.Dense(256, activation = tf.nn.leaky_relu)
        self.dx_dense_2 = tf.keras.layers.Dense(tf.reduce_prod(dx_shape), activation = tf.nn.leaky_relu)
        self.dx_resampled_conv_blocks = [AveragePoolingBlock(pool_size = 2**i, data_format = data_format, filters = self.dx_upsample_channels, use_resnetblocks = True, use_deconv_upsample = True, activation = tf.nn.leaky_relu, kernel_size = 10-2*i) for i in range(4)]
        self.dx_upsample = Upsample([-1, -1], data_format = data_format)

        self.post_dx_conv_blocks = [] #2d convs after merging domain shape info and the result of self.post_conv1d_blocks
        for k in range(self.post_dx_conv_block_number):
            ksize = initial_kernel_size + (k*(final_kernel_size - initial_kernel_size)//(post_dx_conv_block_number-1))
            filters = self.dx_upsample_channels-((k*self.dx_upsample_channels)//(post_dx_conv_block_number))
            block = []
            block.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = ksize, activation = tf.nn.leaky_relu, data_format = self.data_format, padding = 'same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            block.append(ResnetBlock(filters = filters, kernel_size = ksize , activation = tf.nn.leaky_relu, data_format = self.data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            self.post_dx_conv_blocks.append(block)

        if use_batchnorm: #batch norm layers to use if desired
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
        
        self.conv_last = tf.keras.layers.Conv2D(filters = 1, kernel_size = final_kernel_size, activation = tf.nn.tanh, padding = 'same', data_format = data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer) #final convolutions before output
        self.resnet_last = ResnetBlock(filters = 1, kernel_size = final_kernel_size, activation = tf.nn.tanh, data_format = data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        #self.output_denoiser = AveragePoolingBlock(pool_size = 4, data_format=data_format, use_resnetblocks = True, use_deconv_upsample = False, kernel_size = final_kernel_size, filters = 1, activation=tf.nn.tanh, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer) #13.09 set activations to tanh

    def call(self, inp):
       '''
       input arguments:
       
       inp[0]: tf.Tensor of shape (batch_size, 1, y direction dimension) or (batch_size, y direction dimension, 1) based on self.data_format. this must be the BC information.
       inp[1]: tf.Tensor of shape (batch_size, 1). this must be the grid spacing information.
       inp[2]: tf.Tensor of shape (4,) or (). this determines the grid size in the direction perpendicular to the boundary. if not provided, instead the last self.x_output_resolution will be used.
       '''
       self.dx = inp[1] 
       try:
           if inp[2].shape[0] == 1: #set output shape
               self.x_output_resolution = int(inp[2])
           elif self.data_format == 'channels_first':
               self.x_output_resolution = int(inp[2][2])
           else:
               self.x_output_resolution = int(inp[2][1])
       except:
           pass

       out = self.conv1d_layers[0](inp[0]) #1d convs
       for layer in self.conv1d_layers[1:]:
           out = layer(out)

       if self.data_format == 'channels_first': 
           out = tf.expand_dims(out, axis = 1) #convert 1d conv results into 2d
           domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx())], axis = 0)) #calculate domain info
           input_length = inp[0].shape[2] #get mesh size in the direction of the provided BC
       else:
           out = tf.expand_dims(out, axis = -1)
           domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[1]), tf.keras.backend.floatx())], axis = 0))
           input_length = inp[0].shape[1]

       for block in self.post_conv1d_blocks: #2d convs to increase channel no of the 1d conv results
           for layer in block:
               out = layer(out)

       dx_res = self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(domain_info))) #process domain info
       if self.data_format == 'channels_first':
           dx_res = tf.keras.backend.reshape(dx_res, (-1, 1, self.dx_shape[0], self.dx_shape[1]))
       else:
           dx_res = tf.keras.backend.reshape(dx_res, (-1, self.dx_shape[0], self.dx_shape[1], 1))
       dx_upsample_result = self.dx_resampled_conv_blocks[0](dx_res)
       for layer in self.dx_resampled_conv_blocks[1:]:
           dx_upsample_result += layer(dx_res)

       try:#merge domain info
           out = self.dx_upsample([out, [self.x_output_resolution, input_length]]) + self.dx_upsample([dx_upsample_result, [self.x_output_resolution, input_length]])
       except:
           out = self.dx_upsample([out, [64, input_length]]) + self.dx_upsample([dx_upsample_result, [64, input_length]])

       for block_num, block in enumerate(self.post_dx_conv_blocks):#final convolutions to bring channel no to 1
           for layer in block:
               out = layer(out)
           if self.use_batchnorm: #apply batch norm if needed
               out = self.batchnorm_layers[block_num](out, training = self.training)

       if self.use_batchnorm:
           out = self.batchnorm_last(out, training = self.training)

       out = self.conv_last(out)
       out = self.resnet_last(out)
       

       if not self.training:
           out = tf.Variable(lambda: out)
           if self.data_format == 'channels_first':
               out[...,0,:].assign(inp[0])
           else:
               out[:,0,...].assign(inp[0])
       #try:
       #    out = self.output_denoiser(out)
       #except:
       #    print('asd')
       
       return out

    def __call__(self, inp, training = True):#overload __call__ to allow freezing batch norm parameters
        self.training = training
        return super().__call__(inp)

            
        
        
