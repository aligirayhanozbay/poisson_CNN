import tensorflow as tf
import copy

from .resnet import resnet
from ..utils import check_batchnorm_fused_enable, choose_conv_layer, apply_advanced_padding_and_call_conv_layer, get_pooling_method
from ..layers import Upsample, deconvupscale

class bottleneck_block_multilinearupsample(tf.keras.models.Model):
    def __init__(self, ndims, downsampling_factor, filters, conv_kernel_size, data_format = 'channels_first', conv_activation = tf.keras.activations.linear, conv_use_bias = True, use_resnet = False, padding_mode='constant', constant_padding_value=0.0, n_convs = 1, upsampling_factor = None, conv_initializer_constraint_regularizer_options = {}, downsampling_method = 'conv', conv_downsampling_kernel_size = None, pool_downsampling_method = 'max', use_batchnorm = False, batchnorm_trainable = True, resize_method = 'bilinear'):

        super().__init__()

        self.data_format = data_format
        self.downsampling_factor = downsampling_factor
        if upsampling_factor is None:
            upsampling_factor = downsampling_factor
        self.upsampling_factor = upsampling_factor
        self.use_batchnorm = use_batchnorm
        self.downsampling_method = downsampling_method.lower()

        self.filters = filters

        conv_layer = choose_conv_layer(ndims)
        conv_init_args = {'filters': filters, 'kernel_size': conv_kernel_size, 'data_format': self.data_format, 'activation': conv_activation, 'use_bias': conv_use_bias}
        conv_init_args = {**conv_init_args, **conv_initializer_constraint_regularizer_options}
        self.conv_layers = []

        if self.downsampling_method == 'conv':
            downsampling_input_args = copy.deepcopy(conv_init_args)
            downsampling_input_args['padding'] = 'same'
            downsampling_input_args['strides'] = self.downsampling_factor
            downsampling_input_args['kernel_size'] = conv_downsampling_kernel_size
            self.downsample_layer = conv_layer(**downsampling_input_args)
            self._apply_downsample = apply_advanced_padding_and_call_conv_layer(padding_mode, self.downsample_layer, constant_padding_value)
        elif self.downsampling_method == 'pool':
            downsampling_input_args = {'pool_size': downsampling_factor, 'padding':'same', 'data_format':data_format}
            self.downsample_layer = get_pooling_method(pool_downsampling_method, ndims)(**downsampling_input_args)
            if use_resnet:
                first_conv_layer = conv_layer(padding = 'valid', **conv_init_args)
                self.conv_layers.append(first_conv_layer)
            self._apply_downsample = self.downsample_layer
        else:
            raise(ValueError('Downsampling method can only be conv or pool'))

        if use_resnet:
            conv_init_args['ndims'] = ndims
            conv_init_args['use_batchnorm'] = use_batchnorm
            conv_init_args['batchnorm_trainable'] = True
            conv_init_args['padding_mode'] = padding_mode
            conv_init_args['constant_padding_value'] = constant_padding_value
            conv_layer = resnet

        batchorm_fused_enable = check_batchnorm_fused_enable()
        while len(self.conv_layers) < n_convs:
            self.conv_layers.append(conv_layer(**conv_init_args))
            if use_batchnorm and (not use_resnet):
                self.conv_layers.append(tf.keras.layers.BatchNormalization(axis = 1 if data_format == 'channels_first' else -1, trainable = batchnorm_trainable, fused = batchorm_fused_enable))

        self._apply_convolution = []
        for layer in self.conv_layers:
            if isinstance(layer,resnet) or isinstance(layer,tf.keras.layers.BatchNormalization):
                self._apply_convolution.append(layer)
            else:
                self._apply_convolution.append(apply_advanced_padding_and_call_conv_layer(padding_mode,layer,constant_padding_value))
            
        self.upsample_layer = Upsample(ndims = ndims, data_format = self.data_format, resize_method = resize_method)

    #@tf.function
    def call(self, inp):
        conv_inp, domain_sizes = inp

        out = self._apply_downsample(conv_inp)

        for conv_op in self._apply_convolution:
            out = conv_op(out)

        if self.data_format == 'channels_first':
            inpshape = tf.shape(conv_inp)[2:]
        else:
            inpshape = tf.shape(conv_inp)[1:-1]
            
        outshape = tf.cast((inpshape/self.downsampling_factor)*self.upsampling_factor,tf.int32)
        
        out = self.upsample_layer([out, domain_sizes, outshape])

        return out

class bottleneck_block_deconvupsample(bottleneck_block_multilinearupsample):
    def __init__(self, ndims, downsampling_factor, filters, conv_kernel_size, deconv_kernel_size, data_format = 'channels_first', conv_activation = tf.keras.activations.linear, conv_use_bias = True, use_resnet = False, padding_mode='constant', constant_padding_value=0.0, n_convs = 1, upsampling_factor = None, conv_initializer_constraint_regularizer_options = {}, downsampling_method = 'conv', conv_downsampling_kernel_size = None, pool_downsampling_method = 'max', use_batchnorm = False, batchnorm_trainable = True, deconv_activation = tf.keras.activations.linear, deconv_use_bias = True, deconv_initializer_constraint_regularizer_options = {}):

        super().__init__(ndims = ndims, downsampling_factor = downsampling_factor, filters = filters, conv_kernel_size = conv_kernel_size, data_format = data_format, conv_activation = conv_activation, conv_use_bias = conv_use_bias, use_resnet = use_resnet, padding_mode = padding_mode, constant_padding_value = constant_padding_value, n_convs = n_convs, upsampling_factor = upsampling_factor, conv_initializer_constraint_regularizer_options = conv_initializer_constraint_regularizer_options, downsampling_method = downsampling_method, conv_downsampling_kernel_size = conv_downsampling_kernel_size, pool_downsampling_method = pool_downsampling_method, use_batchnorm = use_batchnorm, batchnorm_trainable = batchnorm_trainable)

        deconv_init_args = {'filters':filters, 'kernel_size': deconv_kernel_size, 'upsample_ratio': upsampling_factor, 'data_format':data_format, 'activation': deconv_activation, 'use_bias': deconv_use_bias, 'dimensions': ndims}
        deconv_init_args = {**deconv_init_args, **deconv_initializer_constraint_regularizer_options}
        #import pdb
        #pdb.set_trace()
        self.upsample_layer = deconvupscale(**deconv_init_args)

    @tf.function
    def call(self, inp):

        out = self._apply_downsample(inp)

        for conv_op in self._apply_convolution:
            out = conv_op(out)

        if self.data_format == 'channels_first':
            inpshape = tf.shape(inp)[2:]
            outshape = tf.cast((inpshape/self.downsampling_factor)*self.upsampling_factor,tf.int32)
            outshape = tf.concat([tf.shape(inp)[:1], tf.constant([self.filters]), outshape],0)
        else:
            inpshape = tf.shape(inp)[1:-1]
            outshape = tf.cast((inpshape/self.downsampling_factor)*self.upsampling_factor,tf.int32)
            outshape = tf.concat([tf.shape(inp)[:1], outshape, tf.constant([self.filters])],0)
        
        out = self.upsample_layer([out, outshape])

        return out

if __name__ == '__main__':
    bbma = {
        'ndims': 2,
        'downsampling_factor': 2,
        'filters': 3,
        'conv_kernel_size' : 3,
        'data_format' : 'channels_first',
        'conv_activation' : tf.nn.relu,
        'conv_use_bias' : True,
        'use_resnet' : False,
        'padding_mode':'symmetric',
        'constant_padding_value':0.0,
        'n_convs' : 3,
        'upsampling_factor' : 1.5,
        'conv_initializer_constraint_regularizer_options' : {"kernel_regularizer": tf.keras.regularizers.l2()},
        "downsampling_method":'conv',
        "conv_downsampling_kernel_size": 2,
        "use_batchnorm": True
    }
    bb = bottleneck_block_multilinearupsample(**bbma)
    mod = bb
    mod.compile(loss = 'mse', optimizer=tf.keras.optimizers.Adam())
    
    bsize = 10
    inp = [tf.random.uniform((bsize,1,100,100)), tf.tile(tf.constant([[1.0,1.5]]),[bsize,1])]
    out = tf.random.uniform((bsize,bbma['filters']) + tuple([int(inp[0].shape[k+2]*bbma['upsampling_factor']/bbma['downsampling_factor']) for k in range(2)]))
    
    mod.fit(x=inp, y = out)

    bbda = {
        'ndims': 2,
        'downsampling_factor': 2,
        'filters': 3,
        'conv_kernel_size' : 3,
        'data_format' : 'channels_first',
        'conv_activation' : tf.nn.relu,
        'conv_use_bias' : True,
        'use_resnet' : False,
        'padding_mode':'symmetric',
        'constant_padding_value':0.0,
        'n_convs' : 3,
        'upsampling_factor' : 2,
        'conv_initializer_constraint_regularizer_options' : {"kernel_regularizer": tf.keras.regularizers.l2()},
        "downsampling_method":'conv',
        "conv_downsampling_kernel_size": 2,
        "use_batchnorm": True,
        "deconv_kernel_size": 5,
        "deconv_activation": tf.nn.tanh,
        "deconv_use_bias": True,
        "deconv_initializer_constraint_regularizer_options": {"bias_regularizer": tf.keras.regularizers.l2()}
    }
    bb = bottleneck_block_deconvupsample(**bbda)
    mod = bb
    mod.compile(loss = 'mse', optimizer=tf.keras.optimizers.Adam())
    
    bsize = 10
    inp = tf.random.uniform((bsize,1,100,100))
    out = tf.random.uniform((bsize,bbda['filters']) + tuple([int(inp.shape[k+2]*bbda['upsampling_factor']/bbma['downsampling_factor']) for k in range(2)]))

    mod.fit(x=inp, y = out)
    
    
    import pdb
    pdb.set_trace()
