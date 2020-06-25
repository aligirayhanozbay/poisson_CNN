import tensorflow as tf

from .metalearning_conv import convert_keras_dataformat_to_tf

def convolution_and_bias_add_closure(data_format, conv_method, use_bias, upsample_ratio):
    @tf.function
    def compute_output_shape(newshape, batch_size):
        return tf.tile(tf.expand_dims(tf.concat([tf.constant([1],dtype=tf.int32),newshape[1:]],0),0),[batch_size,1])
        
    if use_bias:
        @tf.function
        def single_sample_upsample(conv_input, kernel, bias, newshape):
            conv_input = tf.expand_dims(conv_input, 0)
            output = conv_method(conv_input, kernel, newshape, padding = "SAME", data_format = data_format, dilations = 1)
            output = tf.nn.bias_add(output, bias, data_format = data_format)
            return output[0,...]

        @tf.function
        def upsample_with_batched_kernels(conv_input, kernels, biases, newshape):
            batch_size = tf.keras.backend.shape(conv_input)[0]
            newshape = compute_output_shape(newshape, batch_size)
            output = tf.map_fn(lambda x: single_sample_upsample(x[0],x[1],x[2],x[3]), [conv_input, kernels, biases, newshape], dtype = tf.keras.backend.floatx())
            return output
    else:
        @tf.function
        def single_sample_upsample(conv_input, kernel, newshape):
            conv_input = tf.expand_dims(conv_input, 0)
            output = conv_method(conv_input, kernel, newshape, padding = "SAME", data_format = data_format, dilations = 1)
            return output[0,...]

        @tf.function
        def upsample_with_batched_kernels(conv_input, kernels, newshape):
            batch_size = tf.keras.backend.shape(conv_input)[0]
            newshape = compute_output_shape(newshape, batch_size)
            output = tf.map_fn(lambda x: single_sample_upsample(x[0],x[1],x[2]), [conv_input, kernels, newshape], dtype = tf.keras.backend.floatx())
            return output
    return upsample_with_batched_kernels


class metalearning_deconvupscale(tf.keras.layers.Layer):
    def __init__(self, upsample_ratio, filters, kernel_size, data_format = 'channels_first', conv_activation = tf.keras.activations.linear, use_bias = True, kernel_initializer = None, bias_initializer = None, kernel_regularizer = None, bias_regularizer = None, activity_regularizer = None, kernel_constraint = None, bias_constraint = None, dimensions = None, dense_activations = tf.keras.activations.linear, pre_output_dense_units = [8,16], **kwargs):
        '''
        An upsampling layer using a transposed convolution the kernel and bias of which is generated with a feedforward NN.
        '''
        
        super().__init__(**kwargs)
        
        if dimensions is None:
            try:
                self.dimensions = len(upsample_ratio)
            except:
                self.dimensions = len(kernel_size)
        else:
            self.dimensions = dimensions

        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size for _ in range(self.dimensions)]
        else:
            self.kernel_size = kernel_size
        
        if isinstance(upsample_ratio, int):
            self.upsample_ratio = [upsample_ratio for _ in range(self.dimensions)]
        else:
            self.upsample_ratio = upsample_ratio
        
        self.conv_activation = conv_activation
        
        self.data_format = data_format
        self._tf_data_format = convert_keras_dataformat_to_tf(self.data_format, self.dimensions)

        self.filters = filters
        
        self.use_bias = use_bias

        self.bias_shape = tf.constant([self.filters])

        if callable(dense_activations):
            self.dense_activations = [dense_activations for _ in range(len(pre_output_dense_units)+1)]
        else:
            self.dense_activations = dense_activations
        

        self._other_dense_layer_args = {'use_bias': use_bias, 'kernel_initializer': kernel_initializer, 'bias_initializer': bias_initializer, 'kernel_regularizer': kernel_regularizer, 'bias_regularizer': bias_regularizer, 'activity_regularizer': activity_regularizer, 'kernel_constraint': kernel_constraint, 'bias_constraint': bias_constraint}
        self.pre_output_dense_units = pre_output_dense_units


        if self.dimensions == 1:
            self.conv_method = lambda *args, **kwargs: tf.nn.conv1d_transpose(*args, strides = self.upsample_ratio, **kwargs)
        elif self.dimensions == 2:
            self.conv_method = lambda *args, **kwargs: tf.nn.conv2d_transpose(*args, strides = self.upsample_ratio, **kwargs)
        elif self.dimensions == 3:
            self.conv_method = lambda *args, **kwargs: tf.nn.conv3d_transpose(*args, strides = self.upsample_ratio, **kwargs)
        else:
            raise(ValueError('dimensions must be 1,2 or 3'))
        
        self.conv_method = convolution_and_bias_add_closure(self._tf_data_format, self.conv_method, self.use_bias, self.upsample_ratio)

    def build(self, input_shape):

        self.previous_layer_filters = input_shape[0][1 if self.data_format == 'channels_first' else -1]
        self.kernel_shape = tf.concat([self.kernel_size,[self.filters],[self.previous_layer_filters]], axis = 0)
        self.bias_shape = tf.constant([self.filters if self.use_bias else 0])
        self.dense_layers = [tf.keras.layers.Dense(self.pre_output_dense_units[k], activation = self.dense_activations[k], **self._other_dense_layer_args) for k in range(len(self.pre_output_dense_units))] + [tf.keras.layers.Dense(tf.reduce_prod(self.kernel_shape)+self.bias_shape, activation = self.dense_activations[-1], **self._other_dense_layer_args)]


    @tf.function
    def call(self, inputs):
        '''
        Perform the 'meta-learning deconv' operation.

        Inputs:
        -inputs: list of 3 tensors [conv_input, dense_input, output_shape]. dense_input is supplied to the dense layers to generate the conv kernel and bias. then the bias, kernel and conv_input and output_shape are supplied to the conv op. output_shape must be supplied as most downsampling ops like pooling map multiple input shapes to a single output shape, and hence the output shape param is necessary to pick the correct output shape.

        Outputs:
        -output: tf.Tensor of shape (batch_size, self.filters, output_shape[2],...,output_spatial_shape[-1]), or the equivalent with channels_last data format
        '''

        #unpack inputs
        conv_input = inputs[0]
        dense_input = inputs[1]
        output_shape = inputs[2]

        #generate conv kernel and bias
        conv_kernel_and_bias = self.dense_layers[0](dense_input)
        for layer in self.dense_layers[1:]:
            conv_kernel_and_bias = layer(conv_kernel_and_bias)

        conv_kernel = tf.reshape(conv_kernel_and_bias[:,:tf.reduce_prod(self.kernel_shape)],tf.concat([[-1],self.kernel_shape], axis = 0))
        
        #perform the convolution and bias addition, apply activation and return
        if self.use_bias:
            bias = conv_kernel_and_bias[:,-tf.squeeze(self.bias_shape):]
            output = self.conv_method(conv_input, conv_kernel, bias, output_shape)
        else:
            output = self.conv_method(conv_input, conv_kernel, output_shape)

        return self.conv_activation(output)

        

    
if __name__ == '__main__':
    
    denseinp = 2*tf.random.uniform((10,5))-1
    convinp = tf.random.uniform((10,2,150,95))
    class TestModel(tf.keras.models.Model):
        def __init__(self):
            super().__init__()

            self.out_channels = 3
            self.pool = tf.keras.layers.MaxPool2D(data_format = 'channels_first',padding='same')
            self.upscale = metalearning_deconvupscale(2, self.out_channels, (13,21), use_bias = True, data_format = 'channels_first')

        #@tf.function
        def call(self, inp):
            conv_input, dense_input = inp
            orig_shape = tf.shape(conv_input)
            if self.upscale.data_format == 'channels_first':
                out_shape = tf.concat([orig_shape[:1],tf.constant([self.out_channels]),orig_shape[2:]],0)
            else:
                out_shape = tf.concat([orig_shape[:1],orig_shape[1:-1],tf.constant([self.out_channels])],0)
            print(out_shape)
            out = self.pool(conv_input)
            print(out.shape)
            out = self.upscale([out, dense_input, out_shape])
            return out


    mod = TestModel()
    print(mod([convinp,denseinp]).shape)
