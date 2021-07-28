import tensorflow as tf

from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow.python.keras import constraints, initializers, regularizers, activations

from .metalearning_conv import convert_keras_dataformat_to_tf

class deconvupscale(tf.keras.layers.Layer):
    def __init__(self, upsample_ratio, filters, kernel_size, data_format = 'channels_first', activation = tf.keras.activations.linear, use_bias = True, kernel_initializer = None, bias_initializer = None, kernel_regularizer = None, bias_regularizer = None, activity_regularizer = None, kernel_constraint = None, bias_constraint = None, dimensions = None):

        super().__init__()
        if dimensions is None:
            try:
                self.dimensions = len(upsample_ratio)
            except:
                self.dimensions = len(kernel_size)
        else:
            self.dimensions = dimensions
            
        self.rank = self.dimensions
        self.filters = filters

        if isinstance(kernel_size, int):
            self.kernel_size = tuple([kernel_size for _ in range(self.dimensions)])
        else:
            self.kernel_size = tuple(kernel_size)

        if isinstance(upsample_ratio, int):
            self.upsample_ratio = tuple([upsample_ratio for _ in range(self.dimensions)])
        else:
            self.upsample_ratio = tuple(upsample_ratio)

        self.data_format = data_format
        self.activation = activation
        self.use_bias = use_bias
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        
        self._tf_data_format = convert_keras_dataformat_to_tf(self.data_format, self.dimensions)

        self.deconv_method = eval('tf.nn.conv' + str(self.dimensions) + 'd_transpose')

    def build(self, input_shape):
        input_shape = input_shape[0]
        input_shape = tf.TensorShape(input_shape)
        if len(input_shape) != (self.dimensions+2):
            raise ValueError('Inputs should have rank '+str(self.rank+2)+'. Received input shape: ' + str(input_shape))
        #channel_axis = self._get_channel_axis()
        channel_axis = 1 if self.data_format == 'channels_first' else -1
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = self.kernel_size + (self.filters, input_dim)

        self.kernel = self.add_weight(name='kernel',shape=kernel_shape,initializer=self.kernel_initializer,regularizer=self.kernel_regularizer,constraint=self.kernel_constraint,trainable=True,dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',shape=(self.filters,),initializer=self.bias_initializer,regularizer=self.bias_regularizer,constraint=self.bias_constraint,trainable=True,dtype=self.dtype)
        else:
            self.bias = None
        self.built = True
        input_spec_shape = [None,input_dim] + [None for _ in range(self.dimensions)] if self.data_format == 'channels_first' else [None] + [None for _ in range(self.dimensions)] + [input_dim]
        self.input_spec = [tf.keras.layers.InputSpec(dtype = self.dtype, shape = input_spec_shape, ndim=self.dimensions+2), tf.keras.layers.InputSpec(dtype = tf.int32, shape=[self.dimensions+2], ndim = 1)]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape[0]).as_list()
        bsize = input_shape[0]
        out_channels = self.filters
        data_dims_shapes = [None for _ in range(self.dimensions)]
                
        if self.data_format == 'channels_first':
            outshape = [bsize,out_channels] + data_dims_shapes
        else:
            outshape = [bsize] + data_dims_shapes + [out_channels]

        return tf.TensorShape(outshape)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'upsample_ratio': self.upsample_ratio,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        return config

    #@tf.function(experimental_relax_shapes=True)
    def call(self, inp):
        conv_inp, output_shape = inp
        
        out = self.deconv_method(conv_inp, self.kernel, output_shape, strides = self.upsample_ratio, padding = "SAME", data_format = self._tf_data_format, dilations = 1)

        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias, data_format = self._tf_data_format)
        
        out = self.activation(out)
        return out

if __name__ == '__main__':
    ds = deconvupscale(upsample_ratio = 2, filters = 2, kernel_size = 5, dimensions = 2)
    print(ds.get_config())
    class dummy_model(tf.keras.models.Model):
        def __init__(self, ds):
            super().__init__()
            self.pool = tf.keras.layers.MaxPool2D(pool_size = 2, data_format = 'channels_first', padding = 'same')
            self.ds = ds
        def call(self, inp):
            out = self.pool(inp[0])
            return self.ds([out,inp[1][0]])
    mod = dummy_model(ds)
    mod.compile(loss='mse',optimizer=tf.keras.optimizers.Adam())
    import numpy as np
    import copy
    class dummy_data_generator(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
        def __len__(self):
            return 50
        def __getitem__(self,idx=0):
            inshape = [10,1,200+int(6*(np.random.rand()-0.5)),200+int(6*(np.random.rand()-0.5))]
            outshape = copy.deepcopy(inshape)
            outshape[1] = 2
            return [tf.random.uniform(inshape), tf.constant([outshape])],tf.random.uniform(outshape)
    mod.fit(dummy_data_generator())
    mod.summary()
        
