import numpy as np
import tensorflow as tf
from collections.abc import Iterable

class Upsample(tf.keras.layers.Layer):
    '''
    Layer that upsamples the input images with one of the upscaling functions provided in tf.image.ResizeMethod module
    '''
    def __init__(self, upsample_ratio = (2,2), data_format = 'channels_first', resize_method = tf.image.ResizeMethod.BICUBIC, align_corners = True, **kwargs):
        super(Upsample, self).__init__(**kwargs)
        #Set output data type
        if tf.keras.backend.floatx() == 'float64':
            self.output_dtype = tf.float64
        elif tf.keras.backend.floatx() == 'float32':
            self.output_dtype = tf.float32
        else:
            self.output_dtype = tf.float16
        self.align_corners = align_corners
        #channels first or channels last
        self.data_format = data_format
        #function to be used from tf.image.ResizeMethod
        self.resize_method = resize_method
        #set upsample ratio
        if not isinstance(upsample_ratio, Iterable):
            self.upsample_ratio = (upsample_ratio, upsample_ratio)
        else:
            self.upsample_ratio = upsample_ratio
            
    def build(self, input_shape):
        super(Upsample, self).build(input_shape)
        
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            return tf.TensorShape(list(input_shape)[0:2] + [self.upsample_ratio[0] * input_shape[-2], self.upsample_ratio[1] * input_shape[-1]])
        else:
            return tf.TensorShape([input_shape[0], self.upsample_ratio[0] * input_shape[-3], self.upsample_ratio[1] * input_shape[-2], input_shape[-1]])
        
    def call(self, inputs):
        if self.data_format == 'channels_first':
            return tf.cast(tf.transpose(tf.image.resize_images(tf.transpose(inputs, (0,2,3,1)), tf.multiply(self.upsample_ratio, inputs.shape[2:]), align_corners=self.align_corners, method=self.resize_method), (0,3,1,2)), self.output_dtype)
        else:
            return tf.cast(tf.image.resize_images(inputs, tf.multiply(self.upsample_ratio, inputs.shape[1:-1]), align_corners=self.align_corners, method=self.resize_method), self.output_dtype)
        