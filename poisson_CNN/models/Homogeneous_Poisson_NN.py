import tensorflow as tf
import copy

from ..layers import metalearning_conv, metalearning_deconvupscale
from ..blocks import bottleneck_block, metalearning_resnet

def get_init_arguments_from_config(cfg,k,fields_in_cfg,fields_in_args):
    args = copy.deepcopy(cfg)

    for field_in_cfg,field_in_args in zip(fields_in_cfg,fields_in_args):
        del args[field_in_cfg]
        args[field_in_args] = cfg[field_in_cfg][k]
    
    return args

    

class Homogeneous_Poisson_NN(tf.keras.models.Model):
    def __init__(self, ndims, data_format = 'channels_first', final_convolutions_config = None, pre_bottleneck_convolutions_config = None, bottleneck_config = None, use_batchnorm = False):

        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.data_format = data_format

        if pre_bottleneck_convolutions_config is None:
            pre_bottleneck_convolutions_config= {
                'filters': [8,16,24,32],
                'kernel_sizes': [19,17,15,13],
                'padding_mode': 'SYMMETRIC',
                'conv_activation': tf.nn.leaky_relu,
                'dense_activations': tf.nn.leaky_relu,
                'use_bias': True,
                'pre_output_dense_units': [8,16]
                }
        

        fields_in_conv_cfg = ['filters','kernel_sizes']
        fields_in_conv_args = ['filters', 'kernel_size']
        prev_layer_filters_initialconv = [1] + pre_bottleneck_convolutions_config['filters'][:-1]
        self.pre_bottleneck_convolutions = []
        for k in range(len(pre_bottleneck_convolutions_config['filters'])):#
            conv_args = get_init_arguments_from_config(pre_bottleneck_convolutions_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv = metalearning_conv(previous_layer_filters = prev_layer_filters_initialconv[k], data_format = data_format, dimensions = ndims, padding = 'same', **conv_args)
            self.pre_bottleneck_convolutions.append(conv)
            if self.use_batchnorm:
                self.pre_bottleneck_convolutions.append(tf.keras.layers.BatchNormalization(axis = 1 if self.data_format == 'channels_first' else -1))

        if bottleneck_config is None:
            n_bottleneck_layers = 6
            bottleneck_config = {
                'downsampling_factors': [2**(k) for k in range(n_bottleneck_layers)],
                'upsampling_factors': [2**(k) for k in range(n_bottleneck_layers)],
                'filters': pre_bottleneck_convolutions_config['filters'][-1],
                'conv_kernel_sizes': [13,13,13,13,11,7],
                'deconv_kernel_sizes': [2**(k+1) for k in range(n_bottleneck_layers)],
                'n_convs': [2 for _ in range(n_bottleneck_layers)],
                'conv_padding_mode': 'SYMMETRIC',
                'conv_conv_activation': tf.nn.leaky_relu,
                'conv_dense_activation': tf.nn.leaky_relu,
                'conv_pre_output_dense_units': [8,16],
                'conv_use_bias': True,
                'deconv_conv_activation': tf.nn.leaky_relu,
                'deconv_dense_activation': tf.nn.leaky_relu,
                'deconv_pre_output_dense_units': [8,16],
                'deconv_use_bias': True,
                'use_resnet': True,
                'conv_downsampling_kernel_sizes': [2**(k) for k in range(n_bottleneck_layers)],
            }
        
        fields_in_bottleneck_cfg = ['downsampling_factors', 'upsampling_factors', 'conv_kernel_sizes', 'deconv_kernel_sizes', 'n_convs', 'conv_downsampling_kernel_sizes']
        fields_in_bottleneck_args = ['downsampling_factor', 'upsampling_factor', 'conv_kernel_size', 'deconv_kernel_size', 'n_convs', 'conv_downsampling_kernel_size']
        prev_layer_filters_bottleneck = pre_bottleneck_convolutions_config['filters'][-1]
        self.bottleneck_blocks = [bottleneck_block(ndims = ndims, data_format = data_format, previous_layer_filters = prev_layer_filters_bottleneck, use_batchnorm = self.use_batchnorm, **get_init_arguments_from_config(bottleneck_config,k,fields_in_bottleneck_cfg,fields_in_bottleneck_args)) for k in range(len(bottleneck_config['downsampling_factors']))]

        if final_convolutions_config is None:
            final_convolutions_config = {
                'filters': [bottleneck_config['filters']] + [16,8,4,1],
                'kernel_sizes': [7,5,3,3,3],
                'padding_mode': 'SYMMETRIC',
                'conv_activation': tf.nn.leaky_relu,
                'dense_activations': tf.nn.leaky_relu,
                'use_bias': True,
                'pre_output_dense_units': [8,16]
                }
        self.final_convolutions = []
        prev_layer_filters_finalconv = [bottleneck_config['filters']] + final_convolutions_config['filters'][:-1]
        print(prev_layer_filters_finalconv)
        final_convolution_stages = len(final_convolutions_config['filters'])
        for k in range(final_convolution_stages-2):
            conv_args = get_init_arguments_from_config(final_convolutions_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv = metalearning_resnet(previous_layer_filters = prev_layer_filters_finalconv[k], use_batchnorm = self.use_batchnorm, data_format = data_format, dimensions = ndims, **conv_args)
            self.final_convolutions.append(conv)
        for k in range(final_convolution_stages-2,final_convolution_stages):
            conv_args = get_init_arguments_from_config(final_convolutions_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv = metalearning_conv(previous_layer_filters = prev_layer_filters_finalconv[k], data_format = data_format, dimensions = ndims, padding = 'same', **conv_args)
            self.final_convolutions.append(conv)


    @tf.function
    def call(self,inp):
        conv_inp, dense_inp = inp

        initial_conv_result = self.pre_bottleneck_convolutions[0]([conv_inp,dense_inp])
        for layer in self.pre_bottleneck_convolutions[1:]:
            initial_conv_result = layer([initial_conv_result, dense_inp])

        bottleneck_result = self.bottleneck_blocks[0]([initial_conv_result,dense_inp])
        for layer in self.bottleneck_blocks[1:]:
            bottleneck_result = bottleneck_result + layer([initial_conv_result,dense_inp])

        out = self.final_convolutions[0]([bottleneck_result, dense_inp])
        for layer in self.final_convolutions[1:]:
            out = layer([out,dense_inp])

        return out
        
        
if __name__ == '__main__':
    mod = Homogeneous_Poisson_NN(2, use_batchnorm = False)
    convinp = 2*tf.random.uniform((1,1,500,500))-1
    denseinp = tf.random.uniform((1,6))

    print(mod([convinp,denseinp]).shape)
    import time
    ntrials = 15
    t0 = time.time()
    for k in range(ntrials):
        print(mod([convinp,denseinp]).shape)
    t1 = time.time()
    print((t1-t0)/ntrials)
    import pdb
    pdb.set_trace()
        

        

        


        
