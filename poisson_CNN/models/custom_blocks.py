import tensorflow as tf
import itertools
import opt_einsum as oe
import numpy as np
import copy

from ..layers import Upsample, DeconvUpscale2D, SpatialPyramidPool

class SepConvBlock(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', separable_kernel_size  = (5,256), nonsep_kernel_size = 5, separable_activation = tf.nn.leaky_relu, nonsep_activation = tf.nn.leaky_relu, separable_filters = 8, nonsep_filters = 4):
        super().__init__()
        self.separableconv2d = tf.keras.layers.SeparableConv2D(separable_filters, kernel_size = separable_kernel_size, padding = 'same', activation = separable_activation, data_format = data_format)
        self.conv2d = tf.keras.layers.Conv2D(filters = nonsep_filters, kernel_size = nonsep_kernel_size, activation = nonsep_activation, padding = 'same', data_format = data_format)
        
    def call(self, inp):
        return self.conv2d(self.separableconv2d(inp))
    
class ResnetBlock(tf.keras.models.Model):
    def __init__(self, ndims = 2, data_format = 'channels_first', **conv_layer_args):
        super().__init__()
        if int(ndims) == 1:
            self.conv0 = tf.keras.layers.Conv1D(padding = 'same', data_format = data_format, **conv_layer_args)
            self.conv1 = tf.keras.layers.Conv1D(padding = 'same', data_format = data_format, **conv_layer_args)
        elif int(ndims) == 2:
            self.conv0 = tf.keras.layers.Conv2D(padding = 'same', data_format = data_format, **conv_layer_args)
            self.conv1 = tf.keras.layers.Conv2D(padding = 'same', data_format = data_format, **conv_layer_args)
        elif int(ndims) == 3:
            self.conv0 = tf.keras.layers.Conv3D(padding = 'same', data_format = data_format, **conv_layer_args)
            self.conv1 = tf.keras.layers.Conv3D(padding = 'same', data_format = data_format, **conv_layer_args)
        else:
            raise ValueError('ndims must be 1,2 or 3')

    def call(self, inp):
        return inp + self.conv0(self.conv1(inp))
    
class ResampledConvolutionBlock(tf.keras.models.Model):
    def __init__(self, resample_arg, data_format = 'channels_first', resize_method = tf.image.ResizeMethod.BICUBIC, use_resnetblocks = True, pre_upsample_conv_blocks = 2, post_upsample_conv_blocks = 2, **convblockargs):
        super().__init__()
        self.data_format = data_format
        self.resample = Upsample(resample_arg, data_format = data_format, resize_method = resize_method)
        self.resize_to_original_shape = Upsample([-1, -1], data_format = data_format)
        if use_resnetblocks:
            self.pre_upsample_conv_blocks = [ResnetBlock(data_format = data_format, **convblockargs) for i in range(pre_upsample_conv_blocks)]
            self.post_upsample_conv_blocks = [ResnetBlock(data_format = data_format, **convblockargs) for i in range(post_upsample_conv_blocks)]
        else:
            self.pre_upsample_conv_blocks = [tf.keras.layers.Conv2D(data_format = data_format, padding = 'same', **convblockargs) for i in range(pre_upsample_conv_blocks)]
            self.post_upsample_conv_blocks = [tf.keras.layers.Conv2D(data_format = data_format, padding = 'same', **convblockargs) for i in range(post_upsample_conv_blocks)]
        
    def call(self, inp):
        if self.data_format == 'channels_first':
            original_shape = [inp.shape[2], inp.shape[3]]
        else:
            original_shape = [inp.shape[1], inp.shape[2]]
        out = self.resample(inp)
        for convblock in self.pre_upsample_conv_blocks:
            out = convblock(out)
        out = self.resize_to_original_shape([out, original_shape])
        for convblock in self.post_upsample_conv_blocks:
            out = convblock(out)
        return out

    
    
class AveragePoolingBlock(tf.keras.models.Model):
    def __init__(self, pool_size = 2, data_format = 'channels_first', resize_method = tf.image.ResizeMethod.BICUBIC, use_resnetblocks = False, use_deconv_upsample = False, **convblockargs):
        super().__init__()
        self.data_format = data_format
        self.pool = tf.keras.layers.AveragePooling2D(data_format = data_format, pool_size = pool_size, padding = 'same')
        self.use_deconv_upsample = use_deconv_upsample
        
        if not use_deconv_upsample:
            self.upsample = Upsample([-1,-1],resize_method=resize_method, data_format = data_format)
        else:
            deconvblockargs = copy.deepcopy(convblockargs)
            #_ = deconvblockargs.pop('kernel_regularizer', None) #backprop has difficulties with regularizing deconv weights so remove these.
            #_ = deconvblockargs.pop('bias_regularizer', None)
            self.upsample = DeconvUpscale2D(pool_size, data_format = data_format, **deconvblockargs)

        if not use_resnetblocks:
            self.pooledconv = tf.keras.layers.Conv2D(data_format = data_format, padding='same', **convblockargs)
            self.upsampledconv = tf.keras.layers.Conv2D(data_format = data_format, padding='same', **convblockargs)
        else:
            self.pooledconv = ResnetBlock(data_format = data_format, **convblockargs)
            self.upsampledconv = ResnetBlock(data_format = data_format, **convblockargs)

    def call(self, inp):
        if self.data_format == 'channels_first':
            input_shape = [inp.shape[-2], inp.shape[-1]]
        else:
            input_shape = [inp.shape[-3],inp.shape[-2]]
        return self.upsampledconv(self.upsample([self.pooledconv(self.pool(inp)), input_shape]))
        

class Scaling(tf.keras.models.Model):
    def __init__(self, stages = 2, downsampling_ratio_per_stage = 2, data_format = 'channels_first', padding = 'same', spp_levels = [[2,2], 3, 5], **convargs):
        super().__init__()
        self.spp = SpatialPyramidPool(spp_levels, data_format = data_format, pooling_type = 'MAX')
        self.stages = []
        self.data_format = data_format
        for k in range(stages):
            self.stages.append(tf.keras.layers.Conv2D(data_format = data_format, padding = padding, **convargs))
            self.stages.append(tf.keras.layers.AveragePooling2D(pool_size = downsampling_ratio_per_stage, padding = padding, data_format = data_format))

        self.dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dense_1 = tf.keras.layers.Dense(25, activation = tf.nn.leaky_relu)
        self.dense_2 = tf.keras.layers.Dense(1, activation = 'linear')

    def call(self, inp):
        #input format: tf.Tensor of shape [batch_size, 2, nx, ny] where [:,0,...] is the input to conditionally scale, or a list of 2 tf.Tensors such that [input_to_scale, 2nd input]
        if self.data_format == 'channels_first' and len(inp) == 3:
            conv_data = tf.concat(inp[:2], axis = 1)
            inp_to_scale = tf.expand_dims(conv_data[:,0,...], axis = 1)
        elif self.data_format == 'channels_first':
            conv_data = inp[0]
            inp_to_scale = tf.expand_dims(inp[0][:,0,...], axis = 1)
        elif len(inp) == 3:
            conv_data = tf.concat(inp[:2], axis = -1)
            inp_to_scale = tf.expand_dims(conv_data[:,...,0], axis = -1)
        else:
            conv_data = inp[0]
            inp_to_scale = tf.expand_dims(inp[:,...,0], axis = -1)
        out = self.stages[0](conv_data)
        for k in range(1, len(self.stages)):
            out = self.stages[k](out)
        out = self.spp(out)
        #out = tf.concat([out, inp[1]], axis = 1)
        out = self.dense_2(self.dense_1(self.dense_0(out)))
        return tf.einsum('i...,i...->i...', 1.0+out, inp_to_scale)
        
        
    
