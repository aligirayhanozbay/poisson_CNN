import tensorflow as tf
import copy

class DeconvUpscale2D(tf.keras.layers.Layer):
    def __init__(self, upsample_ratio, kernel_size, activation = tf.nn.leaky_relu, use_bias = True, kernel_initializer = 'glorot_uniform', bias_initializer = 'zeros', kernel_regularizer = None, bias_regularizer = None, data_format = 'channels_first', kernel_constraint = None, bias_constraint = None, input_channels = None, filters = None, **kwargs):

        self.data_format = data_format
        # if data_format == 'channels_first':
        #     self.data_format = 'NCHW'
        # else:
        #     self.data_format = 'NHWC'

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        
        self.use_bias = use_bias
        self.activation = activation

        self.kernel_size = kernel_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size for k in range(2)]
        
        self.upsample_ratio = upsample_ratio
        
        self.built = False
        
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.data_format == 'channels_first':#if self.data_format[1] == 'C':
            input_channels = input_shape[1]
        else:
            input_channels = input_shape[-1]
        
        self.kernel = self.add_weight(name = 'kernel', shape = tuple(list(self.kernel_size) + [input_channels, input_channels]), trainable = True, dtype = tf.keras.backend.floatx(), initializer = self.kernel_initializer, regularizer = self.kernel_regularizer, constraint = self.kernel_constraint)
        self.bias = self.add_weight(name = 'bias', shape = (input_channels,), trainable = True, dtype = tf.keras.backend.floatx(), initializer = self.bias_initializer, regularizer = self.bias_regularizer, constraint = self.bias_constraint)

        super().build(input_shape)

    def call(self, inp):
        output_shape = list(inp[1])
        if len(output_shape) != len(inp[0].shape):
            if self.data_format == 'channels_first':#if self.data_format[1] == 'C':
                output_shape = list(inp[0].shape)[:2] + output_shape
            else:
                input_shape = list(inp[0].shape)
                output_shape = [input_shape[0]] + output_shape + [input_shape[-1]]
        input_tensor = inp[0]
        #out = tf.nn.conv2d_transpose(inp[0], strides = self.upsample_ratio, filters = self.kernel, data_format = self.data_format, output_shape = output_shape, padding = 'SAME')
        #out = tf.nn.bias_add(out, self.bias, data_format = self.data_format)
        out = tf.keras.backend.conv2d_transpose(inp[0], self.kernel, strides = (self.upsample_ratio, self.upsample_ratio), data_format = self.data_format, output_shape = output_shape, padding = 'same')
        out = tf.keras.backend.bias_add(out, self.bias, data_format = self.data_format)
        return self.activation(out)
    
    def __call__(self, inp):
        if self.built == False:
            self.built = True
            self.build(inp[0].shape)
        return super().__call__(inp)
