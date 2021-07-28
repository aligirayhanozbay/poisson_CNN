import tensorflow as tf
import copy

from ..layers import metalearning_conv, metalearning_deconvupscale, Upsample
from ..utils import get_pooling_method, check_batchnorm_fused_enable
from .metalearning_resnet import metalearning_resnet

class metalearning_bottleneck_block_deconvupsample(tf.keras.models.Model):
    def __init__(self, ndims, downsampling_factor, filters, conv_kernel_size, deconv_kernel_size, n_convs = 1, conv_padding_mode = 'constant', conv_constant_padding_value = 0.0, conv_conv_activation = tf.keras.activations.linear, conv_dense_activation = tf.keras.activations.linear, conv_pre_output_dense_units = [8,16], conv_use_bias = True, deconv_conv_activation = tf.keras.activations.linear, deconv_dense_activation = tf.keras.activations.linear, deconv_pre_output_dense_units = [8,16], deconv_use_bias = True, use_resnet = False, upsampling_factor = None, data_format = 'channels_first', conv_initializer_constraint_regularizer_options = {}, deconv_initializer_constraint_regularizer_options = {}, downsampling_method = 'conv', conv_downsampling_kernel_size = None, pool_downsampling_method = 'max', use_batchnorm = False, batchnorm_trainable = True):
        '''
        A block which downsamples the output by downsampling_factor, applies a convolution operation and upsamples.

        Inputs:
        -ndims: int. Number of spatial dimensions, can be 1 2 or 3.
        -downsampling_factor: int or list of ints. Each dimension in the downsampled resolution will have dimension[k]//downsampling_factor[k] size.
        -filters: int. The output will have this many channels.
        -conv_kernel_size: int or list of ints. The shape of the kernel of the downsampled convolution operation(s) will be this.
        -deconv_kernel_size: int or list of ints. The The shape of the kernel of the upsampling transposed convolution operation(s) will be this.
        -n_convs: int. Number of convolution operations on the downsampled image.
        -conv_padding_mode: string. see tf.pad mode argument.
        -conv_constant_padding_value: float. determines tf.pad constant_values argument for conv layers.
        -conv_conv_activation: callable. the activation function for the convolution operation(s) on the downsampled image.
        -conv_dense_activation: callable or list of callables with identical size as conv_pre_output_dense_units. the activation functions for the feedforward NNs generating the conv kernels for the convs on the downsampled image.
        -conv_pre_output_dense_units: list of ints. number of dense units in the intermediate layers of the feedforward NNs generating the conv kernels for the convs on the downsampled image.
        -conv_use_bias: bool. determines if bias should be used in the conv operations on the downsampled image.
        -deconv_conv_activation: callable. the activation function for the transposed convolution operation upsampling the downsampled image.
        -deconv_dense_activation: callable or list of callables with identical size as conv_pre_output_dense_units. the activation functions for the feedforward NNs generating the conv kernels for the transposed convolution operation upsampling the downsampled image.
        -deconv_pre_output_dense_units: callable or list of callables with identical size as deconv_pre_output_dense_units. the activation functions for the feedforward NN generating the conv kernels for the transposed convolution operation upsampling the downsampled image.
        -deconv_use_bias: bool. determines if bias should be used in the transposed conv operation upsampling the downsampled image.
        -use_resnet: bool. if enabled, all conv operations on the downsampled images will be resnet blocks.
        -upsampling_factor: int or list of ints. determines the upsampling factor for the transposed conv operation. if set to None, the default value will be chosen as identical to the downsampling_factor argument so the original shape is recovered. when manually supplied, each spatial dim in the output will be input_dim[k]/downsampling_factor*upsampling_factor
        -data_format: str. see keras docs.
        -conv_initializer_constraint_regularizer_option: dict. provide kernel_initializer, kernel_constraint etc arguments in this dict for the conv layers.
        -deconv_initializer_constraint_regularizer_option: dict. provide kernel_initializer, kernel_constraint etc arguments in this dict for the deconv layer
        -downsampling_method: str, 'conv' or 'pool'. use a strided convolution to do downsampling if 'conv' is chosen and pooling if 'pool' is chosen.
        -conv_downsampling_kernel_size: int. kernel size for the conv downsampling method. if not supplied and 'conv' downsampling is chosen, it will be set to conv_kernel_size.
        -pool_downsampling_method: str, 'max' or 'average'. if pool downsampling is chosen, determines if max or average pooling should be used.
        -use_batchnorm: bool. If set to true, a batchnorm layer is added after the transposed conv.
        -batchnorm_trainable: bool. If set to true, the batchnorm layer will be trainable (see keras docs).
        '''

        super().__init__()
        
        self.data_format = data_format
        self.downsampling_factor = downsampling_factor
        if upsampling_factor is None:
            upsampling_factor = downsampling_factor
        self.upsampling_factor = upsampling_factor

        self.filters = filters

        self.conv_layers = []
            
        conv_input_args = {'dimensions': ndims, 'filters': filters, 'kernel_size': conv_kernel_size, 'strides': None, 'dilation_rate': None, 'padding': 'same', 'padding_mode': conv_padding_mode, 'constant_padding_value': conv_constant_padding_value, 'data_format': data_format, 'conv_activation': conv_conv_activation, 'dense_activations': conv_dense_activation, 'pre_output_dense_units': conv_pre_output_dense_units, 'use_bias': conv_use_bias, 'use_batchnorm': use_batchnorm}
        if use_resnet:
            conv_input_args['use_batchnorm'] = use_batchnorm

        self.downsampling_method = downsampling_method
        if downsampling_method == 'conv':
            if conv_downsampling_kernel_size is None:
                conv_downsampling_kernel_size = conv_kernel_size
            downsampling_input_args = {'dimensions':ndims, 'filters': filters, 'kernel_size': conv_downsampling_kernel_size, 'strides': downsampling_factor, 'dilation_rate': None, 'padding': 'same', 'padding_mode': conv_padding_mode, 'constant_padding_value': conv_constant_padding_value, 'data_format': data_format, 'conv_activation': conv_conv_activation, 'dense_activations': conv_dense_activation, 'pre_output_dense_units': conv_pre_output_dense_units, 'use_bias': conv_use_bias}
            downsampling_input_args = {**downsampling_input_args, **conv_initializer_constraint_regularizer_options}#merges dicts
            self.downsample_layer = metalearning_conv(**downsampling_input_args)
        elif downsampling_method == 'pool':
            downsampling_input_args = {'pool_size': downsampling_factor, 'padding':'same', 'data_format':data_format}
            self.downsample_layer = get_pooling_method(pool_downsampling_method, ndims)(**downsampling_input_args)
            first_conv_layer_input_args = copy.deepcopy(conv_input_args)
            if use_resnet:
                del first_conv_layer_input_args['use_batchnorm']
            first_conv_layer = metalearning_conv(**first_conv_layer_input_args)
            self.conv_layers.append(first_conv_layer)

        if use_resnet:
            del conv_input_args['padding']
        else:
            del conv_input_args['use_batchnorm']

        conv_layer = metalearning_resnet if use_resnet else metalearning_conv
            
        while len(self.conv_layers) < n_convs:
            self.conv_layers.append(conv_layer(**conv_input_args))

        upsampling_input_args = {'upsample_ratio': upsampling_factor, 'filters': filters, 'kernel_size': deconv_kernel_size, 'data_format': data_format, 'conv_activation': deconv_conv_activation, 'dense_activations': deconv_dense_activation, 'use_bias': deconv_use_bias, 'dimensions': ndims, 'pre_output_dense_units': deconv_pre_output_dense_units}
        upsampling_input_args = {**upsampling_input_args, **deconv_initializer_constraint_regularizer_options}

        self.upsample_layer = metalearning_deconvupscale(**upsampling_input_args)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            batchnorm_fused_enable = check_batchnorm_fused_enable()
            self.batchnorm = tf.keras.layers.BatchNormalization(axis = 1 if self.data_format == 'channels_first' else -1, trainable = batchnorm_trainable, fused = batchnorm_fused_enable)

            
    @tf.function
    def call(self,inp):
        conv_inp, dense_inp = inp
        
        if self.downsampling_method == 'conv':
            out = self.downsample_layer([conv_inp,dense_inp])
        else:
            out = self.downsample_layer(conv_inp)

        for layer in self.conv_layers:
            out = layer([out,dense_inp])
            
        if self.data_format == 'channels_first':
            inpshape = tf.shape(conv_inp)[2:]
            outshape = tf.cast((inpshape/self.downsampling_factor)*self.upsampling_factor,tf.int32)
            outshape = tf.concat([tf.shape(conv_inp)[:1], tf.constant([self.filters]), outshape],0)
        else:
            inpshape = tf.shape(conv_inp)[1:-1]
            outshape = tf.cast((inpshape/self.downsampling_factor)*self.upsampling_factor,tf.int32)
            outshape = tf.concat([tf.shape(conv_inp)[:1], outshape, tf.constant([self.filters])],0)

        out = self.upsample_layer([out,dense_inp,outshape])
        out = tf.reshape(out,outshape)
        if self.use_batchnorm:
            out = self.batchnorm(out)
        return out

class metalearning_bottleneck_block_multilinearupsample(tf.keras.models.Model):
    def __init__(self, ndims, downsampling_factor, filters, conv_kernel_size, n_convs = 1, conv_padding_mode = 'constant', conv_constant_padding_value = 0.0, conv_conv_activation = tf.keras.activations.linear, conv_dense_activation = tf.keras.activations.linear, conv_pre_output_dense_units = [8,16], conv_use_bias = True, use_resnet = False, upsampling_factor = None, data_format = 'channels_first', conv_initializer_constraint_regularizer_options = {}, downsampling_method = 'conv', conv_downsampling_kernel_size = None, pool_downsampling_method = 'max', use_batchnorm = False, batchnorm_trainable = True):
        super().__init__()
        
        self.data_format = data_format
        self.downsampling_factor = downsampling_factor
        if upsampling_factor is None:
            upsampling_factor = downsampling_factor
        self.upsampling_factor = upsampling_factor

        self.filters = filters

        self.conv_layers = []

        conv_input_args = {'dimensions': ndims, 'filters': filters, 'kernel_size': conv_kernel_size, 'strides': None, 'dilation_rate': None, 'padding': 'same', 'padding_mode': conv_padding_mode, 'constant_padding_value': conv_constant_padding_value, 'data_format': data_format, 'conv_activation': conv_conv_activation, 'dense_activations': conv_dense_activation, 'pre_output_dense_units': conv_pre_output_dense_units, 'use_bias': conv_use_bias, 'use_batchnorm': use_batchnorm}
        if use_resnet:
            conv_input_args['use_batchnorm'] = use_batchnorm
        
        self.downsampling_method = downsampling_method
        if downsampling_method == 'conv':
            if conv_downsampling_kernel_size is None:
                conv_downsampling_kernel_size = conv_kernel_size
            downsampling_input_args = {'dimensions':ndims, 'filters': filters, 'kernel_size': conv_downsampling_kernel_size, 'strides': downsampling_factor, 'dilation_rate': None, 'padding': 'same', 'padding_mode': conv_padding_mode, 'constant_padding_value': conv_constant_padding_value, 'data_format': data_format, 'conv_activation': conv_conv_activation, 'dense_activations': conv_dense_activation, 'pre_output_dense_units': conv_pre_output_dense_units, 'use_bias': conv_use_bias}
            downsampling_input_args = {**downsampling_input_args, **conv_initializer_constraint_regularizer_options}#merges dicts
            self.downsample_layer = metalearning_conv(**downsampling_input_args)
        elif downsampling_method == 'pool':
            downsampling_input_args = {'pool_size': downsampling_factor, 'padding':'same', 'data_format':data_format}
            self.downsample_layer = get_pooling_method(pool_downsampling_method, ndims)(**downsampling_input_args)
            first_conv_layer_input_args = copy.deepcopy(conv_input_args)
            if use_resnet:
                del first_conv_layer_input_args['use_batchnorm']
            first_conv_layer = metalearning_conv(**first_conv_layer_input_args)
            self.conv_layers.append(first_conv_layer)

        if use_resnet:
            del conv_input_args['padding']
        else:
            del conv_input_args['use_batchnorm']

        conv_layer = metalearning_resnet if use_resnet else metalearning_conv
            
        while len(self.conv_layers) < n_convs:
            self.conv_layers.append(conv_layer(**conv_input_args))
            if use_batchnorm and (not use_resnet):
               self.conv_layers.append(tf.keras.layers.BatchNormalization(axis = 1 if self.data_format == 'channels_first' else -1, trainable = batchnorm_trainable))

        self.upsample_layer = Upsample(ndims=ndims, data_format = self.data_format)

    @tf.function
    def call(self, inp):
        conv_inp, dense_inp, domain_sizes = inp

        if self.downsampling_method == 'conv':
            out = self.downsample_layer([conv_inp,dense_inp])
        else:
            out = self.downsample_layer(conv_inp)

        for layer in self.conv_layers:
            out = layer([out,dense_inp])
            
        if self.data_format == 'channels_first':
            inpshape = tf.shape(conv_inp)[2:]
            outshape = tf.cast((inpshape/self.downsampling_factor)*self.upsampling_factor,tf.int32)
            #outshape = tf.concat([tf.shape(conv_inp)[:1], tf.constant([self.filters]), outshape],0)
        else:
            inpshape = tf.shape(conv_inp)[1:-1]
            outshape = tf.cast((inpshape/self.downsampling_factor)*self.upsampling_factor,tf.int32)

        out = self.upsample_layer([out,domain_sizes,outshape])
        #out = tf.reshape(out,outshape)
        return out
        

    
if __name__ == '__main__':
    ndims = 2
    n_prevch = 2
    n_postch = 4
    bsize = 10
    spatial_shape = [2500 for _ in range(ndims)]
    n_dense_features = 10
    conv_inp = tf.random.uniform([bsize,n_prevch] + spatial_shape)
    dense_inp = tf.random.uniform((bsize,n_dense_features))

    downsampling_factor = [6,4]
    upsampling_factor = [3,3]
    conv_ksize = 6
    deconv_ksize = 6
    n_convs = 3
    conv_padding_mode = 'SYMMETRIC'
    conv_activations = tf.nn.leaky_relu
    dense_activations = tf.nn.leaky_relu
    use_batchnorm = True
    bot = metalearning_bottleneck_block_deconvupsample(ndims,downsampling_factor,n_postch,conv_ksize,deconv_ksize,n_convs = n_convs, conv_padding_mode = conv_padding_mode,conv_conv_activation = conv_activations, conv_dense_activation = dense_activations,use_resnet = True,data_format = 'channels_first',upsampling_factor = upsampling_factor, use_batchnorm = use_batchnorm)
    

    class dummy_data_generator(tf.keras.utils.Sequence):
        def __init__(self,batch_size,n_prevch,n_postch,spatial_shape,downsampling_factor,upsampling_factor,n_dense_features):
            self.inshape = tf.constant([batch_size,n_prevch] + spatial_shape)
            self.outshape = tf.constant([batch_size,n_postch] + [int(spatial_shape[k]*upsampling_factor[k]/downsampling_factor[k]) for k in range(len(spatial_shape))])
            self.dense_shape = tf.constant([batch_size,n_dense_features])
        def __getitem__(self,idx=0):
            return [tf.random.uniform(self.inshape),tf.random.uniform(self.dense_shape)], tf.random.uniform(self.outshape)
        def __len__(self):
            return 50

    conv = metalearning_conv(filters=2,kernel_size=3,padding='same',dimensions=ndims)
    class dummy_model(tf.keras.models.Model):
        def __init__(self,conv,bottleneck):
            super().__init__()
            self.conv = conv
            self.bot = bottleneck
        def call(self,inp):
            ci,di = inp
            out = self.conv([ci,di])
            return self.bot([out,di])

    dg = dummy_data_generator(bsize,n_prevch,n_postch,spatial_shape,downsampling_factor,upsampling_factor,n_dense_features)

    mod = dummy_model(conv,bot)
    mod.compile(loss='mse',optimizer=tf.keras.optimizers.Adam())
    mod.fit(dg,epochs=5)
            
    
    '''
    res = bot([conv_inp,dense_inp])
    print(res.shape)
    import time
    t0 = time.time()
    ntrials = 50
    for k in range(ntrials):
        print(bot([conv_inp,dense_inp]).shape)
    t1 = time.time()
    print((t1-t0)/ntrials)
    '''

        

        

        
