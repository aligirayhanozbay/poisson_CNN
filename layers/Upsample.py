import numpy as np
try:
    import tensorflow.compat.v1 as tf
except:
    import tensorflow as tf
from collections.abc import Iterable

class Upsample(tf.keras.layers.Layer):
    '''
    Layer that upsamples the input images with one of the upscaling functions provided in tf.image.ResizeMethod module
    '''
    def __init__(self, dim_args, data_format = 'channels_first', resize_method = tf.image.ResizeMethod.BICUBIC, align_corners = True, **kwargs):
        super().__init__(**kwargs)
        for val in dim_args:
            if not (isinstance(val,float) or isinstance(val,int)):
                raise(TypeError('Supply floats or ints as the dimension arguments!'))
        
        self.align_corners = align_corners
        #channels first or channels last
        self.data_format = data_format
        #function to be used from tf.image.ResizeMethod
        self.resize_method = resize_method
        #set upsample ratio
        self.dim_args = dim_args #floats will scale that dimension to int(old_dim * dim_args[i]). ints will scale straight to that size.
        #self.first_call = True
        if self.data_format == 'channels_first':
            self.data_dims = [2,3]
        elif self.data_format == 'channels_last':
            self.data_dims = [1,2]
            
    def build(self, input_shape):
        super().build(input_shape)
    
    def get_newshape(self, inp_shape):
        new_shape = [inp_shape[i] for i in self.data_dims]
        for i in range(len(self.dim_args)): 
            if isinstance(self.dim_args[i],int):
                new_shape[i] = self.dim_args[i]
            elif isinstance(self.dim_args[i],float):
                if inp_shape[self.data_dims[i]] == None:
                    new_shape[i] = None
                else:
                    new_shape[i] = int(inp_shape[self.data_dims[i]] * self.dim_args[i])
        return new_shape
    def compute_output_shape(self, input_shape):
        new_shape = self.get_newshape(input_shape)
        if self.data_format == 'channels_first':
            return tf.TensorShape(list(input_shape)[0:2] + new_shape)
        else:
            return tf.TensorShape([input_shape[0]] + new_shape + [input_shape[-1]])
        
    def call(self, inputs):
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            set_unknown_dims_as = inputs[1]
            inputs = inputs[0]
            newshape = self.get_newshape(inputs.shape)
            if -1 in newshape:
                newshape = [newshape[i] if newshape[i] != -1 else set_unknown_dims_as[i] for i in range(len(newshape))] 
        else:
            newshape = self.get_newshape(inputs.shape)
        if self.data_format == 'channels_first':
            return tf.cast(tf.transpose(tf.image.resize_images(tf.transpose(inputs, (0,2,3,1)), newshape, align_corners=self.align_corners, method=self.resize_method), (0,3,1,2)), tf.keras.backend.floatx())
        else:
            return tf.cast(tf.image.resize_images(inputs, newshape, align_corners=self.align_corners, method=self.resize_method), tf.keras.backend.floatx())
        