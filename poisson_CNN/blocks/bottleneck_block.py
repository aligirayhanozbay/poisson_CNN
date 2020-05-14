import tensorflow as tf
import copy

from ..layers import metalearning_conv, metalearning_deconvupscale
from .metalearning_resnet import metalearning_resnet

def get_pooling_method(pool_downsampling_method, ndims):
    pool_downsampling_method = pool_downsampling_method[0].upper() + pool_downsampling_method[1:].lower()
    pooling_layer_name = 'tf.keras.layers.' + pool_downsampling_method + 'Pooling' + str(ndims) + 'D'
    return eval(pooling_layer_name)

class bottleneck_block(tf.keras.models.Model):
    def __init__(self, ndims, downsampling_factor, previous_layer_filters, filters, conv_kernel_size, deconv_kernel_size, n_convs = 1, conv_padding_mode = 'constant', conv_constant_padding_value = 0.0, conv_conv_activation = tf.keras.activations.linear, conv_dense_activation = tf.keras.activations.linear, conv_pre_output_dense_units = [8,16], conv_use_bias = True, deconv_conv_activation = tf.keras.activations.linear, deconv_dense_activation = tf.keras.activations.linear, deconv_pre_output_dense_units = [8,16], deconv_use_bias = True, use_resnet = False, upsampling_factor = None, data_format = 'channels_first', conv_initializer_constraint_regularizer_options = {}, deconv_initializer_constraint_regularizer_options = {}, downsampling_method = 'conv', pool_downsampling_method = 'max'):

        super().__init__()
        
        self.data_format = data_format
        self.downsampling_factor = downsampling_factor
        if upsampling_factor is None:
            upsampling_factor = downsampling_factor
        self.upsampling_factor = upsampling_factor

        self.previous_layer_filters = previous_layer_filters
        self.filters = filters

        self.conv_layers = []
            
        conv_input_args = {'dimensions': ndims, 'previous_layer_filters': filters, 'filters': filters, 'kernel_size': conv_kernel_size, 'strides': None, 'dilation_rate': None, 'padding': 'same', 'padding_mode': conv_padding_mode, 'constant_padding_value': conv_constant_padding_value, 'data_format': data_format, 'conv_activation': conv_conv_activation, 'dense_activations': conv_dense_activation, 'pre_output_dense_units': conv_pre_output_dense_units, 'use_bias': conv_use_bias}
        
        if downsampling_method == 'conv':
            downsampling_input_args = {'dimensions':ndims,'previous_layer_filters': previous_layer_filters, 'filters': filters, 'kernel_size': conv_kernel_size, 'strides': downsampling_factor, 'dilation_rate': None, 'padding': 'same', 'padding_mode': conv_padding_mode, 'constant_padding_value': conv_constant_padding_value, 'data_format': data_format, 'conv_activation': conv_conv_activation, 'dense_activations': conv_dense_activation, 'pre_output_dense_units': conv_pre_output_dense_units, 'use_bias': conv_use_bias}
            downsampling_input_args = {**downsampling_input_args, **conv_initializer_constraint_regularizer_options}#merges dicts
            self.downsample_layer = metalearning_conv(**downsampling_input_args)
        elif downsampling_method == 'pool':
            downsampling_input_args = {'pool_size': downsampling_factor, 'padding':'same', 'data_format':data_format}
            self.downsample_layer = get_pooling_method(pool_downsampling_method, ndims)(**downsampling_input_args)
            if previous_layer_filters != filters:
                first_conv_layer_input_args = copy.deepcopy(conv_input_args)
                first_conv_layer_input_args['previous_layer_filters'] = previous_layer_filters
                first_conv_layer = metalearning_conv(**first_conv_layer_input_args)
                self.conv_layers.append(first_conv_layer)

        if use_resnet:
            del conv_input_args['padding']

        conv_layer = metalearning_resnet if use_resnet else metalearning_conv
            
        while len(self.conv_layers) < n_convs:
            self.conv_layers.append(conv_layer(**conv_input_args))

        upsampling_input_args = {'upsample_ratio': upsampling_factor, 'previous_layer_filters': filters, 'filters': filters, 'kernel_size': deconv_kernel_size, 'data_format': data_format, 'conv_activation': deconv_conv_activation, 'dense_activations': deconv_dense_activation, 'use_bias': deconv_use_bias, 'dimensions': ndims, 'pre_output_dense_units': deconv_pre_output_dense_units}
        upsampling_input_args = {**upsampling_input_args, **deconv_initializer_constraint_regularizer_options}

        self.upsample_layer = metalearning_deconvupscale(**upsampling_input_args)

    @tf.function
    def call(self,inp):
        conv_inp, dense_inp = inp

        out = self.downsample_layer([conv_inp,dense_inp])

        for layer in self.conv_layers:
            out = layer([out,dense_inp])

        if self.data_format == 'channels_first':
            inpshape = tf.shape(conv_inp)[2:]
            outshape = tf.cast((inpshape/self.downsampling_factor)*self.upsampling_factor,tf.int32)
            outshape = tf.concat([tf.shape(conv_inp)[:1], tf.constant([self.filters]), outshape],0)
        else:
            inpshape = tf.shape(conv_inp)[1:-1]
            outshape = tf.cast((inpshape/self.downsampling_factor)*self.upsampling_factor,tf.int32)

        out = self.upsample_layer([out,dense_inp,outshape])

        return out

if __name__ == '__main__':
    ndims = 2
    n_prevch = 2
    n_postch = 4
    bsize = 10
    conv_inp = tf.random.uniform([bsize,n_prevch] + [250 for _ in range(ndims)])
    dense_inp = tf.random.uniform((10,10))

    downsampling_factor = 2
    upsampling_factor = 3
    conv_ksize = 5
    deconv_ksize = 5
    n_convs = 3
    conv_padding_mode = 'SYMMETRIC'
    conv_activations = tf.nn.leaky_relu
    dense_activations = tf.nn.leaky_relu
    mod = bottleneck_block(ndims,downsampling_factor,n_prevch,n_postch,conv_ksize,deconv_ksize,n_convs = n_convs, conv_padding_mode = conv_padding_mode,conv_conv_activation = conv_activations, conv_dense_activation = dense_activations,use_resnet = True,data_format = 'channels_first',upsampling_factor = upsampling_factor)
    res = mod([conv_inp,dense_inp])
    print(res.shape)
    import time
    t0 = time.time()
    ntrials = 50
    for k in range(ntrials):
        print(mod([conv_inp,dense_inp]).shape)
    t1 = time.time()
    print((t1-t0)/ntrials)

        

        

        
