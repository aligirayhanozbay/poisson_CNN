import tensorflow as tf
import itertools
import opt_einsum as oe
import numpy as np

from ..layers import Upsample, DeconvUpscale2D

class SepConvBlock(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', separable_kernel_size  = (5,256), nonsep_kernel_size = 5, separable_activation = tf.nn.leaky_relu, nonsep_activation = tf.nn.leaky_relu, separable_filters = 8, nonsep_filters = 4):
        super().__init__()
        self.separableconv2d = tf.keras.layers.SeparableConv2D(separable_filters, kernel_size = separable_kernel_size, padding = 'same', activation = separable_activation, data_format = data_format)
        self.conv2d = tf.keras.layers.Conv2D(filters = nonsep_filters, kernel_size = nonsep_kernel_size, activation = nonsep_activation, padding = 'same', data_format = data_format)
        
    def call(self, inp):
        return self.conv2d(self.separableconv2d(inp))
    
class ResnetBlock(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', **conv_layer_args):
        super().__init__()
        self.conv0 = tf.keras.layers.Conv2D(padding = 'same', data_format = data_format, **conv_layer_args)
        self.conv1 = tf.keras.layers.Conv2D(padding = 'same', data_format = data_format, **conv_layer_args)

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
            self.upsample = DeconvUpscale2D(pool_size, data_format = data_format, **convblockargs)

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
        
