import tensorflow as tf

from .metalearning_conv import convert_keras_dataformat_to_tf
    
def convolution_and_bias_add_closure(data_format, conv_method, use_bias, upsample_ratio):
    @tf.function
    def compute_output_shape(original_shape, kernel_shape):
        if data_format == 'channels_first' or data_format[1] == 'C':
            return tf.concat([original_shape[:1],kernel_shape[-2:-1],original_shape[2:]*upsample_ratio],0)
        elif data_format == 'channels_last' or data_format[-1] == 'C':
            return tf.concat([original_shape[:1],original_shape[1:-1]*upsample_ratio,kernel_shape[-2:-1]],0)
        
    if use_bias:
        @tf.function
        def single_sample_upsample(conv_input, kernel, bias):
            conv_input = tf.expand_dims(conv_input, 0)
            newshape = compute_output_shape(tf.shape(conv_input), tf.shape(kernel))
            output = conv_method(conv_input, kernel, newshape, padding = "SAME", data_format = data_format, dilations = 1)
            output = tf.nn.bias_add(output, bias, data_format = data_format)
            return output[0,...]

        @tf.function
        def upsample_with_batched_kernels(conv_input, kernels, biases):
            batch_size = tf.keras.backend.shape(conv_input)[0]
            output = tf.map_fn(lambda x: single_sample_upsample(x[0],x[1],x[2]), [conv_input, kernels, biases], dtype = tf.keras.backend.floatx())
            return output
    else:
        @tf.function
        def single_sample_upsample(conv_input, kernel):
            conv_input = tf.expand_dims(conv_input, 0)
            newshape = compute_output_shape(tf.shape(conv_input), tf.shape(kernel))
            output = conv_method(conv_input, kernel, newshape, padding = "SAME", data_format = data_format, dilations = 1)
            return output[0,...]

        @tf.function
        def upsample_with_batched_kernels(conv_input, kernels):
            batch_size = tf.keras.backend.shape(conv_input)[0]
            output = tf.map_fn(lambda x: single_sample_upsample(x[0],x[1]), [conv_input, kernels], dtype = tf.keras.backend.floatx())
            return output
    return upsample_with_batched_kernels

class metalearning_deconvupscale(tf.keras.models.Model):
    def __init__(self, upsample_ratio, previous_layer_filters, filters, kernel_size, data_format = 'channels_first', conv_activation = tf.keras.activations.linear, use_bias = True, kernel_initializer = None, bias_initializer = None, kernel_regularizer = None, bias_regularizer = None, activity_regularizer = None, kernel_constraint = None, bias_constraint = None, dimensions = None, dense_activations = tf.keras.activations.linear, pre_output_dense_units = [8,16], **kwargs):
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
        self.previous_layer_filters = previous_layer_filters
        
        self.use_bias = use_bias

        self.kernel_shape = tf.concat([self.kernel_size,[self.filters],[self.previous_layer_filters]], axis = 0)
        self.bias_shape = tf.constant([self.filters])

        if callable(dense_activations):
            dense_activations = [dense_activations for _ in range(len(pre_output_dense_units)+1)]

        dense_layer_args = {'use_bias': use_bias, 'kernel_initializer': kernel_initializer, 'bias_initializer': bias_initializer, 'kernel_regularizer': kernel_regularizer, 'bias_regularizer': bias_regularizer, 'activity_regularizer': activity_regularizer, 'kernel_constraint': kernel_constraint, 'bias_constraint': bias_constraint}
            
        self.dense_layers = [tf.keras.layers.Dense(pre_output_dense_units[k], activation = dense_activations[k], **dense_layer_args) for k in range(len(pre_output_dense_units))] + [tf.keras.layers.Dense(tf.reduce_prod(self.kernel_shape)+self.bias_shape, activation = dense_activations[-1], **dense_layer_args)]

        if self.dimensions == 1:
            self.conv_method = lambda *args, **kwargs: tf.nn.conv1d_transpose(*args, strides = self.upsample_ratio, **kwargs)
        elif self.dimensions == 2:
            self.conv_method = lambda *args, **kwargs: tf.nn.conv2d_transpose(*args, strides = self.upsample_ratio, **kwargs)
        elif self.dimensions == 3:
            self.conv_method = lambda *args, **kwargs: tf.nn.conv3d_transpose(*args, strides = self.upsample_ratio, **kwargs)
        else:
            raise(ValueError('dimensions must be 1,2 or 3'))
        
        self.conv_method = convolution_and_bias_add_closure(self._tf_data_format, self.conv_method, self.use_bias, self.upsample_ratio)

    @tf.function
    def call(self, inputs):
        '''
        Perform the 'meta-learning conv' operation.

        Inputs:
        -inputs: list of 2 tensors [conv_input, dense_input]. dense_input is supplied to the dense layers to generate the conv kernel and bias. then the bias, kernel and conv_input are supplied to the conv op.

        Outputs:
        -output: tf.Tensor of shape (batch_size, self.filters, output_spatial_shape_1,...,output_spatial_shape_dimensions)
        '''

        #unpack inputs
        conv_input = inputs[0]
        dense_input = inputs[1]

        #generate conv kernel and bias
        conv_kernel_and_bias = self.dense_layers[0](dense_input)
        for layer in self.dense_layers[1:]:
            conv_kernel_and_bias = layer(conv_kernel_and_bias)

        conv_kernel = tf.reshape(conv_kernel_and_bias[:,:tf.reduce_prod(self.kernel_shape)],tf.concat([[conv_kernel_and_bias.shape[0]],self.kernel_shape], axis = 0))

        #perform the convolution and bias addition, apply activation and return
        if self.use_bias:
            bias = conv_kernel_and_bias[:,-tf.squeeze(self.bias_shape):]
            output = self.conv_method(conv_input, conv_kernel, bias)
        else:
            output = self.conv_method(conv_input, conv_kernel)

        return self.conv_activation(output)

        

    
if __name__ == '__main__':
    '''
    conv_method = lambda *args, **kwargs: tf.nn.conv2d_transpose(*args, strides = (2,2), **kwargs)
    conv_method = convolution_and_bias_add_closure('NCHW', conv_method, use_bias = False, upsample_ratio = (2,2))
    inp = tf.random.uniform((10,1,100,100))
    kernels = tf.random.uniform((10,5,5,3,1))
    print(conv_method(inp,kernels).shape)

    conv_method = lambda *args, **kwargs: tf.nn.conv2d_transpose(*args, strides = (2,2), **kwargs)
    conv_method = convolution_and_bias_add_closure('NCHW', conv_method, use_bias = True, upsample_ratio = (2,2))
    inp = tf.random.uniform((10,1,100,100))
    kernels = tf.random.uniform((10,5,5,3,1))
    biases = tf.random.uniform((10,3))
    print(conv_method(inp,kernels,biases).shape)
    '''

    mod = metalearning_deconvupscale(2, 2, 3, (5,5))
    convinp = tf.random.uniform((10,2,100,100))
    denseinp = 2*tf.random.uniform((10,5))-1
    res = mod([convinp,denseinp])
    print(res.shape)
    import time
    t0 = time.time()
    ntrials = 100
    for k in range(ntrials):
        q = mod([convinp+k/100,denseinp+k/100])
    print((time.time()-t0)/ntrials)
    
