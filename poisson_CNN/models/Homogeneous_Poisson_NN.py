import tensorflow as tf
import copy
import math

from .Homogeneous_Poisson_NN_Metalearning import get_init_arguments_from_config, process_normalizations, process_output_scaling_modes, process_regularizer_initializer_and_constraint_arguments
from ..layers import deconvupscale
from ..blocks import bottleneck_block_multilinearupsample, bottleneck_block_deconvupsample, resnet
from ..dataset.utils import set_max_magnitude_in_batch_and_return_scaling_factors, set_max_magnitude_in_batch, build_fd_coefficients
from ..utils import apply_advanced_padding_and_call_conv_layer, choose_conv_layer, check_batchnorm_fused_enable

class Homogeneous_Poisson_NN(tf.keras.models.Model):
    def __init__(self, ndims, data_format = 'channels_first', final_convolutions_config = None, pre_bottleneck_convolutions_config = None, bottleneck_upsampling = 'deconv', bottleneck_config = None, use_batchnorm = False, input_normalization = None, output_scaling = None):

        super().__init__()
        self.ndims = ndims

        self.input_normalization = process_normalizations(input_normalization)
        self._input_normalization_has_to_be_performed = tf.reduce_any([bool(x) for x in self.input_normalization.values()])
        self.output_scaling = process_output_scaling_modes(output_scaling)
        #self._output_scaling_has_to_be_performed = tf.reduce_any(list(self.output_scaling.values()))
        if self.output_scaling['match_peak_laplacian_magnitude_to_peak_rhs']:
            self.fd_stencil = tf.constant(poisson_CNN.dataset.utils.build_fd_coefficients(5, 2, ndims), dtype=tf.keras.backend.floatx())
            self.fd_conv_method = choose_conv_method(ndims)
        
        self.use_batchnorm = use_batchnorm
        self.data_format = data_format
        self.bottleneck_upsampling = bottleneck_upsampling

        if pre_bottleneck_convolutions_config is None:
            raise(ValueError('Provide a config for pre bottleneck convolutions'))
        if bottleneck_config is None:
            raise(ValueError('Provide a config for bottleneck blocks'))
        if final_convolutions_config is None:
            raise(ValueError('Provide a config for final convolutions'))

        conv_layer = choose_conv_layer(ndims)

        #initial convolutions
        fields_in_conv_cfg = ['filters','kernel_sizes']
        fields_in_conv_args = ['filters', 'kernel_size']
        self.pre_bottleneck_convolutions = []#contains the layers
        self.pre_bottleneck_convolution_ops = []#contains functions that implement more advanced padding and then does convolution
        pre_bottleneck_convolutions_config = copy.deepcopy(pre_bottleneck_convolutions_config)
        pre_bottleneck_convolutions_padding_mode = pre_bottleneck_convolutions_config.pop('padding_mode','CONSTANT')
        pre_bottleneck_convolutions_constant_padding_value = pre_bottleneck_convolutions_config.pop('constant_padding_value',0.0)
        for k in range(len(pre_bottleneck_convolutions_config['filters'])):#
            conv_args = get_init_arguments_from_config(pre_bottleneck_convolutions_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv = conv_layer(data_format = data_format, padding = 'valid', **conv_args)
            self.pre_bottleneck_convolutions.append(conv)
            self.pre_bottleneck_convolution_ops.append(apply_advanced_padding_and_call_conv_layer(pre_bottleneck_convolutions_padding_mode, self.pre_bottleneck_convolutions[-1], constant_padding_value = pre_bottleneck_convolutions_constant_padding_value))
            if self.use_batchnorm:
                batchnorm_fused_enable = check_batchnorm_fused_enable()
                bnorm = tf.keras.layers.BatchNormalization(axis = 1 if self.data_format == 'channels_first' else -1, fused = batchnorm_fused_enable)
                self.pre_bottleneck_convolutions.append(bnorm)
                self.pre_bottleneck_convolution_ops.append(bnorm)


        #bottleneck blocks
        if bottleneck_upsampling == 'deconv':
            fields_in_bottleneck_cfg = ['downsampling_factors', 'upsampling_factors', 'conv_kernel_sizes', 'deconv_kernel_sizes', 'n_convs'] + (['conv_downsampling_kernel_sizes'] if ('conv_downsampling_kernel_sizes' in bottleneck_deconv_config.keys()) else [])
            fields_in_bottleneck_args = ['downsampling_factor', 'upsampling_factor', 'conv_kernel_size', 'deconv_kernel_size', 'n_convs'] + (['conv_downsampling_kernel_size'] if ('conv_downsampling_kernel_sizes' in bottleneck_deconv_config.keys()) else [])
            self.bottleneck_blocks = [bottleneck_block_deconvupsample(ndims = ndims, data_format = data_format, use_batchnorm = self.use_batchnorm, **get_init_arguments_from_config(bottleneck_config,k,fields_in_bottleneck_cfg,fields_in_bottleneck_args)) for k in range(len(bottleneck_config['downsampling_factors']))]
        elif bottleneck_upsampling == 'multilinear':
            fields_in_bottleneck_cfg = ['downsampling_factors', 'upsampling_factors', 'conv_kernel_sizes', 'n_convs'] + (['conv_downsampling_kernel_sizes'] if ('conv_downsampling_kernel_sizes' in bottleneck_deconv_config.keys()) else [])
            fields_in_bottleneck_args = ['downsampling_factor', 'upsampling_factor', 'conv_kernel_size', 'n_convs'] + (['conv_downsampling_kernel_size'] if ('conv_downsampling_kernel_sizes' in bottleneck_deconv_config.keys()) else [])
            self.bottleneck_blocks = [bottleneck_block_multilinearupsample(ndims = ndims, data_format = data_format, use_batchnorm = self.use_batchnorm, **get_init_arguments_from_config(bottleneck_config,k,fields_in_bottleneck_cfg,fields_in_bottleneck_args)) for k in range(len(bottleneck_config['downsampling_factors']))]
        else:
            raise(ValueError('Invalid bottleneck block upsampling method'))
        self.n_bottleneck_blocks = tf.cast(len(self.bottleneck_blocks),tf.keras.backend.floatx())
        self.bottleneck_blocks = sorted(self.bottleneck_blocks, key = lambda x: x.downsampling_factor, reverse=True)

        #final convolutions
        self.final_convolutions = []
        self.final_convolution_ops = []
        final_convolution_stages = len(final_convolutions_config['filters'])
        final_convolutions_padding_mode = final_convolutions_config.pop('padding_mode','CONSTANT')
        final_convolutions_constant_padding_value = final_convolutions_config.pop('constant_padding_value',0.0)
        for k in range(final_convolution_stages):
            conv_args = get_init_arguments_from_config(final_convolutions_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv = conv_layer(data_format = data_format, padding = 'valid', **conv_args)
            self.final_convolutions.append(conv)
            self.final_convolution_ops.append(apply_advanced_padding_and_call_conv_layer(final_convolutions_padding_mode, self.final_convolutions[-1], constant_padding_value = final_convolutions_constant_padding_value))
            if self.use_batchnorm and (k<(final_convolution_stages-1)):
                bnorm = tf.keras.layers.BatchNormalization(axis = 1 if self.data_format == 'channels_first' else -1, fused = batchnorm_fused_enable)
                self.pre_bottleneck_convolutions.append(bnorm)
                self.pre_bottleneck_convolution_ops.append(bnorm)

    @tf.function
    def compute_domain_sizes(self, dx, domain_shape):
        domain_sizes = tf.einsum('ij,j->ij', dx, tf.cast(domain_shape-1,dx.dtype))
        return domain_sizes
    '''
    @tf.function
    def rhs_max_magnitude_input_normalization(self,rhses):
    rhses, scaling_factors = set_max_magnitude_in_batch(rhses,self.input_normalization['rhs_max_magnitude'],return_scaling_factors = True)
        return rhses, scaling_factors
    '''

    @tf.function
    def normalize_inputs(self, rhses, dx):
        if self.input_normalization['rhs_max_magnitude'] != False:
            rhses, scaling_factors = set_max_magnitude_in_batch_and_return_scaling_factors(rhses,tf.constant(1.0))
            #rhses, scaling_factors = self.rhs_max_magnitude_input_normalization(rhses)
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
    def scale_outputs(self, rhses, output, max_domain_sizes, grid_spacings):#, input_normalization_factors = None):
        scaling_factors = [tf.ones(tf.shape(rhses)[:1], dtype = rhses.dtype)]
        #if input_normalization_factors is not None:
        #    scaling_factors.append(1/input_normalization_factors)
        if self.output_scaling['match_peak_laplacian_magnitude_to_peak_rhs']:
            scaling_factors.append(self.match_peak_laplacian_magnitude_to_peak_rhs(rhses, output, grid_spacings))
        elif self.output_scaling['soln_max_magnitude']:
            output=set_max_magnitude_in_batch(output,tf.constant(1.0))
            output=tf.reshape(output,tf.shape(rhses))
            return output
        else:
            if self.output_scaling['rhs_max_magnitude']:
                scaling_factors.append(self.rhs_max_magnitude_scaling(rhses))
            if self.output_scaling['max_domain_size_squared']:
                scaling_factors.append(self.max_domain_size_squared_scaling(max_domain_sizes))
        scaling_factors = tf.reduce_prod(tf.stack(scaling_factors,1),1)
        return tf.einsum('i,i...->i...',scaling_factors,output)

    @tf.function
    def generate_position_embeddings(self, batch_size, domain_shape):
        pos_embeddings = tf.stack([tf.broadcast_to(tf.reshape(tf.cos(math.pi * tf.linspace(0.0,1.0,domain_shape[k])),[1 for _ in range(k)] + [-1] + [1 for _ in range(self.ndims-k-1)]), domain_shape) for k in range(self.ndims)],0 if self.data_format == 'channels_first' else -1)
        pos_embeddings = tf.expand_dims(pos_embeddings,0)
        pos_embeddings = tf.tile(pos_embeddings, [batch_size] + [1 for _ in range(self.ndims+1)])
        return pos_embeddings

    @tf.function
    def call(self, inp):

        rhses, dx = inp

        inp_shape = tf.shape(rhses)
        if self.data_format == 'channels_first':
            domain_shape = inp_shape[2:]
        else:
            domain_shape = inp_shape[1:-1]
        batch_size = inp_shape[0]
        domain_sizes = self.compute_domain_sizes(dx, domain_shape)
        max_domain_sizes = tf.reduce_max(domain_sizes,1)
        pos_embeddings = self.generate_position_embeddings(batch_size, domain_shape)

        conv_inp = tf.concat([rhses, pos_embeddings], 1 if self.data_format == 'channels_first' else -1)

        initial_conv_result = self.pre_bottleneck_convolution_ops[0](conv_inp)
        for layer in self.pre_bottleneck_convolution_ops[1:]:
            initial_conv_result = layer(initial_conv_result)
        
        if self.bottleneck_upsampling == 'deconv':
            bottleneck_result = self.bottleneck_blocks[0](initial_conv_result)
            for layer in self.bottleneck_blocks[1:]:
                bottleneck_result = layer(tf.concat([initial_conv_result, bottleneck_result], 1 if self.data_format == 'channels_first' else -1))
        elif self.bottleneck_upsampling == 'multilinear':
            bottleneck_result = self.bottleneck_blocks[0]([initial_conv_result, domain_sizes])
            for layer in self.bottleneck_blocks[1:]:
                bottleneck_result = layer([tf.concat([initial_conv_result, bottleneck_result], 1 if self.data_format == 'channels_first' else -1), domain_sizes])
        #bottleneck_result = bottleneck_result / self.n_bottleneck_blocks

        out = self.final_convolution_ops[0](tf.concat([initial_conv_result, bottleneck_result], 1 if self.data_format == 'channels_first' else -1))
        #out = self.final_convolution_ops[0](bottleneck_result)
        for layer in self.final_convolution_ops[1:]:
            out = layer(out)

        out = self.scale_outputs(rhses,out,max_domain_sizes,dx)

        return out

    def train_step(self,data):

        inputs, ground_truth = data

        rhses, dx = inputs

        #rhses = tf.debugging.check_numerics(rhses, 'nan or inf in rhses. indices: ' + str(tf.where(tf.math.is_nan(rhses))))
        #ground_truth = tf.debugging.check_numerics(ground_truth, 'nan or inf in ground truth')
        #dx = tf.debugging.check_numerics(rhses, 'nan or inf in dxes')
        

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            pred = self(inputs)
            #predk = tf.debugging.check_numerics(pred, 'nan or inf in pred')
            #loss = tf.reduce_mean((ground_truth - pred)**2)
            loss = self.loss_fn(y_true=ground_truth,y_pred=pred,rhs=rhses,dx=dx)
        #lossk = tf.debugging.check_numerics(loss, 'nan or inf in loss')
        grads = tape.gradient(loss,self.trainable_variables)
        '''
        for k in range(len(grads)):
            gradk = tf.debugging.check_numerics(grads[k], 'nan or inf in grad ' + str(k))
            #grads[k] = tf.clip_by_norm(grads[k],tf.constant(0.5))
        '''

        grad_peak_log = tf.constant(-999.0)
        for k in range(len(grads)):
            peak_log = tf.math.log(tf.reduce_max(tf.abs(grads[k])))
            if grad_peak_log < peak_log:
                grad_peak_log = peak_log
                
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        model_peak_log = tf.constant(-999.0)
        for variables in self.trainable_variables:
            var_peak_log = tf.math.log(tf.reduce_max(tf.abs(variables)))
            if model_peak_log < var_peak_log:
                model_peak_log = var_peak_log

        return {'loss' : loss, 'mse': tf.reduce_mean((pred - ground_truth)**2)}# 'peak_rhs' : tf.reduce_max(tf.abs(rhses)), 'peak_soln': tf.reduce_max(tf.abs(ground_truth)), 'peak_pred':tf.reduce_max(tf.abs(pred)), 'model peak log' : model_peak_log, 'grad peak log' : grad_peak_log}

    def compile(self, loss, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss

if __name__ == '__main__':
    input_norm = {'rhs_max_magnitude':True}
    output_scaling = None#{'max_domain_size_squared':True}

    pbcc = {
        "filters": [4,16,32],
        "kernel_sizes": [19,17,15],
        "padding_mode": "symmetric",
        "activation": tf.nn.leaky_relu,
        "use_bias": False,
        "bias_initializer":"zeros"
	}
    bcc = {
        "downsampling_factors": [1,2,3,4,8,16,32,64],
        "upsampling_factors": [1,2,3,4,8,16,32,64],
        "filters": 32,
        "conv_kernel_sizes": [13,13,13,13,13,13,13,13],
        "n_convs": [2,2,2,2,2,2,2,2],
        "padding_mode": "CONSTANT",
        "constant_padding_value": 4.0,
        "conv_activation": tf.nn.leaky_relu,
        "conv_use_bias": False,
        "use_resnet": True,
        "conv_downsampling_kernel_sizes": [3,2,3,4,8,16,32,64],
        "conv_initializer_constraint_regularizer_options":{"kernel_regularizer":tf.keras.regularizers.l2()},
        "downsampling_method": "conv"
	}
    fcc = {
        "filters": [16,12,8,4,2,1],
        "kernel_sizes": [11,7,5,5,3,3],
        "padding_mode": "CONSTANT",
        "constant_padding_value": 2.0,
        "activation": tf.nn.tanh,
        "use_bias": False,
        "bias_initializer":"zeros"
        }

        
    mod = Homogeneous_Poisson_NN(2, use_batchnorm = False, input_normalization = input_norm, output_scaling = output_scaling, pre_bottleneck_convolutions_config = pbcc, bottleneck_config = bcc, final_convolutions_config = fcc, bottleneck_upsampling = 'multilinear')
    convinp = 2*tf.random.uniform((1,1,3000,3000))-1
    denseinp = tf.random.uniform((1,2))

    print(mod([convinp,denseinp]).shape)
    mod.summary()
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
