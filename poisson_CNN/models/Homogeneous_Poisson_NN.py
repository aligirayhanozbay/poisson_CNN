import tensorflow as tf
import copy

from ..layers import metalearning_conv, metalearning_deconvupscale
from ..blocks import bottleneck_block, metalearning_resnet
from ..dataset.utils import set_max_magnitude_in_batch, build_fd_coefficients
from ..dataset.generators.reverse import choose_conv_method

def get_init_arguments_from_config(cfg,k,fields_in_cfg,fields_in_args):
    args = copy.deepcopy(cfg)

    for field_in_cfg,field_in_args in zip(fields_in_cfg,fields_in_args):
        del args[field_in_cfg]
        args[field_in_args] = cfg[field_in_cfg][k]
    
    return args

def process_normalizations(normalizations):
    normalization_types = ['rhs_max_magnitude']
    normalization_default_values = [False]
    if normalizations is None:
        return {key:default_val for key,default_val in zip(normalization_types,normalization_default_values)}
    elif isinstance(normalizations,dict):
        for key,default_val in zip(normalization_types,normalization_default_values):
            if key not in normalizations:
                normalizations[key] = default_val
        if isinstance(normalizations['rhs_max_magnitude'],bool) and normalizations['rhs_max_magnitude']:
            normalizations['rhs_max_magnitude'] = 1.0
    return normalizations

def process_output_scaling_modes(output_scalings):
    output_scaling_modes = ['rhs_max_magnitude', 'max_domain_size_squared', 'match_peak_laplacian_magnitude_to_peak_rhs']
    output_scaling_mode_default_values = [False,False,False]
    if output_scalings is None:
        return {key:default_val for key,default_val in zip(output_scaling_modes,output_scaling_mode_default_values)}
    elif isinstance(output_scalings,dict):
        for key,default_val in zip(output_scaling_modes,output_scaling_mode_default_values):
            if key not in output_scalings:
                output_scalings[key] = default_val
    return output_scalings

class metalearning_conv_and_batchnorm(tf.keras.models.Model):
    def __init__(self,conv,batchnorm):
        super().__init__()
        self.conv = conv
        self.batchnorm = batchnorm

    @tf.function
    def call(self,inp):
        out = self.conv(inp)
        out = self.batchnorm(out)
        return out

@tf.function
def get_peak_magnitudes_in_each_sample(batch):
    return tf.map_fn(lambda x: tf.reduce_max(tf.abs(x)), batch)

class Homogeneous_Poisson_NN(tf.keras.models.Model):
    def __init__(self, ndims, data_format = 'channels_first', final_convolutions_config = None, pre_bottleneck_convolutions_config = None, bottleneck_config = None, use_batchnorm = False, input_normalization = None, output_scaling = None):

        super().__init__()

        self.input_normalization = process_normalizations(input_normalization)
        self._input_normalization_has_to_be_performed = tf.reduce_any([bool(x) for x in self.input_normalization.values()])
        self.output_scaling = process_output_scaling_modes(output_scaling)
        self._output_scaling_has_to_be_performed = tf.reduce_any(list(self.output_scaling.values()))
        if self.output_scaling['match_peak_laplacian_magnitude_to_peak_rhs']:
            self.fd_stencil = tf.constant(poisson_CNN.dataset.utils.build_fd_coefficients(5, 2, ndims), dtype=tf.keras.backend.floatx())
            self.fd_conv_method = choose_conv_method(ndims)
        
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
            #self.pre_bottleneck_convolutions.append(conv)
            if self.use_batchnorm:
                bnorm = tf.keras.layers.BatchNormalization(axis = 1 if self.data_format == 'channels_first' else -1)
                block = metalearning_conv_and_batchnorm(conv,bnorm)
                self.pre_bottleneck_convolutions.append(block)
            else:
                self.pre_bottleneck_convolutions.append(conv)

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
    def compute_domain_sizes(self, dx, domain_shape):
        domain_sizes = tf.einsum('ij,j->ij', dx, tf.cast(domain_shape-1,dx.dtype))
        return domain_sizes

    @tf.function
    def rhs_max_magnitude_input_normalization(self,rhses):
        rhses, scaling_factors = set_max_magnitude_in_batch(rhses,self.input_normalization['rhs_max_magnitude'],return_scaling_factors = True)
        return rhses, scaling_factors

    @tf.function
    def normalize_inputs(self, rhses, dx):
        scaling_factors = tf.ones(rhses.shape[0], dtype=rhses.dtype)
        if self.input_normalization['rhs_max_magnitude'] != False:
            rhses, scaling_factors_rhs_max_magnitude = self.rhs_max_magnitude_input_normalization(rhses)
            scaling_factors *= scaling_factors_rhs_max_magnitude
        return rhses, scaling_factors

    @tf.function
    def rhs_max_magnitude_scaling(self,rhs):
        scaling_factors = tf.map_fn(lambda x: tf.reduce_max(tf.abs(x)), rhs)
        return scaling_factors

    @tf.function
    def max_domain_size_squared_scaling(self, max_domain_sizes):
        return max_domain_sizes**2

    @tf.function
    def match_peak_laplacian_magnitude_to_peak_rhs_scaling(self,rhs,soln,dx):
        kernels = tf.einsum('i...,bi->b...', self.fd_stencil, 1/(dx**2))
        kernels = tf.expand_dims(tf.expand_dims(kernels,-1),-1)
        rhs_computed = tf.map_fn(lambda x: self.fd_conv_method(tf.expand_dims(x[0],0),x[1],data_format=self.data_format), (soln,kernels), dtype=soln.dtype)[:,0,...]
        rhs_peak_magnitudes = get_peak_magnitudes_in_each_sample(rhs)
        rhs_computed_peak_magnitudes = get_peak_magnitudes_in_each_sample(rhs_computed)
        return rhs_peak_magnitudes/rhs_computed_peak_magnitudes

    @tf.function
    def scale_outputs(self, conv_inp, output, max_domain_sizes, grid_spacings, input_normalization_factors = None):
        scaling_factors = []
        if input_normalization_factors is not None:
            scaling_factors.append(1/input_normalization_factors)
        if self.output_scaling['match_peak_laplacian_magnitude_to_peak_rhs'] is False:
            if self.output_scaling['rhs_max_magnitude']:
                scaling_factors.append(self.rhs_max_magnitude_scaling(conv_inp))
            if self.output_scaling['max_domain_size_squared']:
                scaling_factors.append(self.max_domain_size_squared_scaling(max_domain_sizes))
        else:
            scaling_factors.append(self.match_peak_laplacian_magnitude_to_peak_rhs(conv_inp, output, grid_spacings))
            
        scaling_factors = tf.reduce_prod(tf.stack(scaling_factors,1),1)
        return tf.einsum('i,i...->i...',scaling_factors,output)

    @tf.function
    def call(self,inp):

        rhses, dx = inp

        if self.data_format == 'channels_first':
            domain_shape = tf.shape(rhses)[2:]
        else:
            domain_shape = tf.shape(rhses)[1:-1]

        domain_sizes = self.compute_domain_sizes(dx, domain_shape)
        max_domain_sizes = tf.reduce_max(domain_sizes,1)

        dense_inp = tf.concat([dx/domain_sizes,tf.einsum('ij,i->ij',domain_sizes,1/max_domain_sizes)],1)

        if self._input_normalization_has_to_be_performed:
            conv_inp, input_normalization_factors = self.normalize_inputs(rhses, dx)
        else:
            conv_inp = rhses
            input_normalization_factors = tf.ones((rhses.shape[0],),dtype=tf.keras.backend.floatx())

        initial_conv_result = self.pre_bottleneck_convolutions[0]([conv_inp,dense_inp])
        for layer in self.pre_bottleneck_convolutions[1:]:
            initial_conv_result = layer([initial_conv_result, dense_inp])

        bottleneck_result = self.bottleneck_blocks[0]([initial_conv_result,dense_inp])
        for layer in self.bottleneck_blocks[1:]:
            bottleneck_result = bottleneck_result + layer([initial_conv_result,dense_inp])

        out = self.final_convolutions[0]([bottleneck_result, dense_inp])
        for layer in self.final_convolutions[1:]:
            out = layer([out,dense_inp])

        if self._output_scaling_has_to_be_performed:
            out = self.scale_outputs(conv_inp,out,max_domain_sizes,dx,input_normalization_factors)

        return out
        
        
if __name__ == '__main__':
    input_norm = {'rhs_max_magnitude':True}
    output_scaling = {'max_domain_size_squared_scaling':True}
    mod = Homogeneous_Poisson_NN(2, use_batchnorm = True, input_normalization = input_norm, output_scaling = output_scaling)
    convinp = 2*tf.random.uniform((1,1,500,500))-1
    denseinp = tf.random.uniform((1,2))

    print(mod([convinp,denseinp]).shape)
    import time
    ntrials = 15
    t0 = time.time()
    for k in range(ntrials):
        print(mod([convinp,denseinp]).shape)
    t1 = time.time()
    print((t1-t0)/ntrials)

    with tf.GradientTape() as tape:
        tape.watch(mod.trainable_variables)
        out = mod([convinp,denseinp])
    grads = tape.gradient(out,mod.trainable_variables)
    
    import pdb
    pdb.set_trace()
        

        

        


        
