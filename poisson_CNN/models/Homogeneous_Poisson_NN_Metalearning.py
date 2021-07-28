import tensorflow as tf
import copy
import math

from ..layers import metalearning_conv, metalearning_deconvupscale
from ..blocks import metalearning_bottleneck_block_deconvupsample, metalearning_bottleneck_block_multilinearupsample, metalearning_resnet
from ..dataset.utils import set_max_magnitude_in_batch_and_return_scaling_factors, set_max_magnitude_in_batch, build_fd_coefficients
from ..utils import check_batchnorm_fused_enable, choose_conv_layer, choose_conv_method, get_peak_magnitudes_in_each_sample

def get_init_arguments_from_config(cfg,k,fields_in_cfg,fields_in_args):
    '''
    Extracts arguments like this:
    cfg = {'key1': 3, 'key2': [0,1,2,3,4], 'key3': [6,7,8,9,10]}
    k = 2
    fields_in_cfg = ['key2', 'key3']
    fields_in_args = ['key2p', 'key3p']
    => output = {'key1': 3, 'key2p': 2, 'key3p': 8} 

    Inputs:
    -cfg: dict. contains original data.
    -k: int. the output will have the key-val pair fields_in_args[i]:cfg[fields_in_cfg[i][k].
    -fields_in_cfg: list of strings. keys to look for in cfg.
    -fields_in_args : list of strings. fields_in_args[i] will be the key replacing fields_in_cfg[i] in the output.
    '''
    return {**{key:cfg[key] for key in filter(lambda x: x not in fields_in_cfg, cfg.keys())},**{arg_key:cfg[cfg_key][k] for arg_key,cfg_key in zip(fields_in_args, fields_in_cfg)}}

def process_normalizations(normalizations):
    '''
    Processes input normalization modes.
    Inputs:
    -normalizations: dict. If a given normalization type is present with the corresponding value, it is retained. Else, the default value in normalization_default_values is assigned.
    Outputs:
    A dict containing containing key-val pairs normalization_type:enabled_or_not
    '''
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
    output_scalings = copy.deepcopy(output_scalings)
    output_scaling_modes = ['rhs_max_magnitude', 'max_domain_size_squared', 'match_peak_laplacian_magnitude_to_peak_rhs', 'soln_max_magnitude']
    output_scaling_mode_default_values = [False,False,False,False]
    if output_scalings is None:
        return {key:default_val for key,default_val in zip(output_scaling_modes,output_scaling_mode_default_values)}
    elif isinstance(output_scalings,dict):
        for key,default_val in zip(output_scaling_modes,output_scaling_mode_default_values):
            if key not in output_scalings:
                output_scalings[key] = default_val
    return output_scalings

def process_regularizer_initializer_and_constraint_arguments(config_dict):
    special_cases_initializer = ['he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform', 'ones', 'zeros']
    special_cases_regularizer = ['l1', 'l2', 'l1_l2']
    for key,val in zip(config_dict.keys(),config_dict.values()):
        if ('regularizer' in key):
            if val not in special_cases_regularizer:
                config_dict[key] = eval(val)
        elif ('initializer' in key):
            if val not in special_cases_initializer:
                config_dict[key] = eval(val)
        elif ('constraint' in key):
            config_dict[key] = eval(val)

    
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

class Homogeneous_Poisson_NN_Metalearning(tf.keras.models.Model):
    def __init__(self, ndims,  data_format = 'channels_first', final_convolutions_config = None, pre_bottleneck_convolutions_config = None, bottleneck_deconv_config = None, bottleneck_multilinear_config = None, input_normalization = None, output_scaling = None, use_batchnorm = False, postsmoother_iterations = 5):

        super().__init__()

        self.ndims = ndims
        self.input_normalization = process_normalizations(input_normalization)
        self._input_normalization_has_to_be_performed = tf.reduce_any([bool(x) for x in self.input_normalization.values()])
        self.output_scaling = process_output_scaling_modes(output_scaling)
        self._output_scaling_has_to_be_performed = tf.reduce_any(list(self.output_scaling.values()))
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

        #process_regularizer_initializer_and_constraint_arguments(pre_bottleneck_convolutions_config)

        fields_in_conv_cfg = ['filters','kernel_sizes']
        fields_in_conv_args = ['filters', 'kernel_size']
        self.pre_bottleneck_convolutions = []
        for k in range(len(pre_bottleneck_convolutions_config['filters'])):#
            conv_args = get_init_arguments_from_config(pre_bottleneck_convolutions_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv = metalearning_conv(data_format = data_format, dimensions = ndims, padding = 'same', **conv_args)
            #self.pre_bottleneck_convolutions.append(conv)
            #'''
            if self.use_batchnorm:
                batchnorm_fused_enable = check_batchnorm_fused_enable()
                bnorm = tf.keras.layers.BatchNormalization(axis = 1 if self.data_format == 'channels_first' else -1, fused=batchnorm_fused_enable)
                block = metalearning_conv_and_batchnorm(conv,bnorm)
                self.pre_bottleneck_convolutions.append(block)
            else:
                self.pre_bottleneck_convolutions.append(conv)
            #'''

        if bottleneck_upsampling == 'deconv':
            '''
            if "deconv_initializer_constraint_regularizer_options" in bottleneck_config:
                process_regularizer_initializer_and_constraint_arguments(bottleneck_config["deconv_initializer_constraint_regularizer_options"])
            if "conv_initializer_constraint_regularizer_options" in bottleneck_config:
                process_regularizer_initializer_and_constraint_arguments(bottleneck_config["conv_initializer_constraint_regularizer_options"])
            '''

            fields_in_bottleneck_cfg = ['downsampling_factors', 'upsampling_factors', 'conv_kernel_sizes', 'deconv_kernel_sizes', 'n_convs', 'conv_downsampling_kernel_sizes']
            fields_in_bottleneck_args = ['downsampling_factor', 'upsampling_factor', 'conv_kernel_size', 'deconv_kernel_size', 'n_convs', 'conv_downsampling_kernel_size']
            self.bottleneck_blocks = [metalearning_bottleneck_block_deconvupsample(ndims = ndims, data_format = data_format, use_batchnorm = self.use_batchnorm, **get_init_arguments_from_config(bottleneck_config,k,fields_in_bottleneck_cfg,fields_in_bottleneck_args)) for k in range(len(bottleneck_config['downsampling_factors']))]
        elif bottleneck_upsampling == 'multilinear':
            fields_in_bottleneck_cfg = ['downsampling_factors', 'upsampling_factors', 'conv_kernel_sizes', 'n_convs', 'conv_downsampling_kernel_sizes']
            fields_in_bottleneck_args = ['downsampling_factor', 'upsampling_factor', 'conv_kernel_size', 'n_convs', 'conv_downsampling_kernel_size']
            self.bottleneck_blocks = [metalearning_bottleneck_block_multilinearupsample(ndims = ndims, data_format = data_format, use_batchnorm = self.use_batchnorm, **get_init_arguments_from_config(bottleneck_config,k,fields_in_bottleneck_cfg,fields_in_bottleneck_args)) for k in range(len(bottleneck_config['downsampling_factors']))]
        else:
            raise(ValueError('Invalid bottleneck block upsampling method'))
        self.n_bottleneck_blocks = tf.cast(len(self.bottleneck_blocks),tf.keras.backend.floatx())
        self.bottleneck_blocks = sorted(self.bottleneck_blocks, key = lambda x: x.downsampling_factor, reverse=True)

        #process_regularizer_initializer_and_constraint_arguments(final_convolutions_config)
        self.final_convolutions = []
        final_convolution_stages = len(final_convolutions_config['filters'])
        last_two_conv_layers = choose_conv_layer(ndims)
        final_convolutions_config = copy.deepcopy(final_convolutions_config)
        self.final_regular_conv_stages = final_convolutions_config.pop('final_regular_conv_stages',2)
        for k in range(final_convolution_stages-self.final_regular_conv_stages):
            conv_args = get_init_arguments_from_config(final_convolutions_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv_to_adjust_channel_number = metalearning_conv(dimensions = ndims, padding = 'same', data_format = data_format, **conv_args)
            conv = metalearning_resnet(use_batchnorm = self.use_batchnorm, data_format = data_format, dimensions = ndims, **conv_args)
            self.final_convolutions.append(conv_to_adjust_channel_number)
            self.final_convolutions.append(conv)
        for k in range(final_convolution_stages-self.final_regular_conv_stages,final_convolution_stages):
            self.final_convolutions.append(last_two_conv_layers(filters = final_convolutions_config['filters'][k], kernel_size = final_convolutions_config['kernel_sizes'][k],activation = tf.keras.activations.linear, padding='same',data_format=self.data_format,use_bias=final_convolutions_config['use_bias']))
            

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
        scaling_factors = []
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
        linspace_start = tf.constant(0.0, tf.keras.backend.floatx())
        linspace_end = tf.constant(1.0, tf.keras.backend.floatx())
        pos_embeddings = tf.stack([tf.broadcast_to(tf.reshape(tf.cos(math.pi * tf.linspace(linspace_start,linspace_end,domain_shape[k])),[1 for _ in range(k)] + [-1] + [1 for _ in range(self.ndims-k-1)]), domain_shape) for k in range(self.ndims)],0 if self.data_format == 'channels_first' else -1)
        pos_embeddings = tf.expand_dims(pos_embeddings,0)
        pos_embeddings = tf.tile(pos_embeddings, [batch_size] + [1 for _ in range(self.ndims+1)])
        return pos_embeddings

    @tf.function
    def call(self,inp):
        rhses, dx = inp

        inp_shape = tf.shape(rhses)
        if self.data_format == 'channels_first':
            domain_shape = inp_shape[2:]
        else:
            domain_shape = inp_shape[1:-1]
        batch_size = inp_shape[0]
        pos_embeddings = self.generate_position_embeddings(batch_size, domain_shape)

        domain_sizes = self.compute_domain_sizes(dx, domain_shape)
        max_domain_sizes = tf.reduce_max(domain_sizes,1)
        
        dense_inp = tf.concat([dx/domain_sizes,tf.einsum('ij,i->ij',domain_sizes,1/max_domain_sizes)],1)
        conv_inp = tf.concat([rhses, pos_embeddings], 1 if self.data_format == 'channels_first' else -1)
        '''
        if self._input_normalization_has_to_be_performed:
            conv_inp, input_normalization_factors = self.normalize_inputs(rhses, dx)
        else:
            conv_inp = rhses
            input_normalization_factors = tf.ones((rhses.shape[0],),dtype=tf.keras.backend.floatx())
        '''
        initial_conv_result = self.pre_bottleneck_convolutions[0]([conv_inp,dense_inp])
        for layer in self.pre_bottleneck_convolutions[1:]:
            initial_conv_result = layer([initial_conv_result, dense_inp])

        if self.bottleneck_upsampling == 'deconv':
            bottleneck_result = self.bottleneck_blocks[0]([initial_conv_result,dense_inp])
            for layer in self.bottleneck_blocks[1:]:
                bottleneck_result = layer([tf.concat([initial_conv_result, bottleneck_result], 1 if self.data_format == 'channels_first' else -1),dense_inp])
        elif self.bottleneck_upsampling == 'multilinear':
            bottleneck_result = self.bottleneck_blocks[0]([initial_conv_result,dense_inp,domain_sizes])
            for layer in self.bottleneck_blocks[1:]:
                bottleneck_result = layer([tf.concat([initial_conv_result, bottleneck_result], 1 if self.data_format == 'channels_first' else -1),dense_inp,domain_sizes])
        #bottleneck_result = bottleneck_result / self.n_bottleneck_blocks
            
        out = self.final_convolutions[0]([tf.concat([initial_conv_result, bottleneck_result], 1 if self.data_format == 'channels_first' else -1), dense_inp])
        for layer in self.final_convolutions[1:-self.final_regular_conv_stages]:
            out = layer([out,dense_inp])
        for layer in self.final_convolutions[-self.final_regular_conv_stages:]:
            out = layer(out)

        if self._output_scaling_has_to_be_performed:
            out = self.scale_outputs(rhses,out,max_domain_sizes,dx)#,input_normalization_factors)

        return out

    def train_step(self,data):

        inputs, ground_truth = data

        rhses, dx = inputs

        #ground_truth_scaled, ground_truth_scaling_factors = set_max_magnitude_in_batch_and_return_scaling_factors(ground_truth, 1.0)

        #rhses = tf.debugging.check_numerics(rhses, 'nan or inf in rhses. indices: ' + str(tf.where(tf.math.is_nan(rhses))))
        #ground_truth = tf.debugging.check_numerics(ground_truth, 'nan or inf in ground truth')
        #dx = tf.debugging.check_numerics(rhses, 'nan or inf in dxes')
        

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            pred = self(inputs)
            #predk = tf.debugging.check_numerics(pred, 'nan or inf in pred')

            loss = self.loss_fn(y_true=ground_truth,y_pred=pred,rhs=rhses,dx=dx)
            #loss = self.loss_fn(y_true=ground_truth_scaled,y_pred=pred,rhs=tf.einsum('i,i...->i...', ground_truth_scaling_factors, ground_truth_scaled),dx=dx)
        #lossk = tf.debugging.check_numerics(loss, 'nan or inf in loss')
        grads = tape.gradient(loss,self.trainable_variables)
        '''
        for k in range(len(grads)):
            gradk = tf.debugging.check_numerics(grads[k], 'nan or inf in grad ' + str(k))
            #grads[k] = tf.clip_by_norm(grads[k],tf.constant(0.5))
        '''
        
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        return {'loss' : loss, 'mse': tf.reduce_mean((pred - ground_truth)**2)}# 'peak_rhs' : tf.reduce_max(tf.abs(rhses)), 'peak_soln': tf.reduce_max(tf.abs(ground_truth)), 'peak_pred':tf.reduce_max(tf.abs(pred)), 'model peak log' : model_peak_log, 'grad peak log' : grad_peak_log}

    def compile(self, loss, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss

        
        
        
if __name__ == '__main__':
    input_norm = {'rhs_max_magnitude':True}
    output_scaling = {'max_domain_size_squared':True}

    pbcc = {
        "filters": [4,6,8],
        "kernel_sizes": [19,17,15],
        "padding_mode": "CONSTANT",
        "conv_activation": tf.nn.leaky_relu,
        "dense_activations": ["tanh","tanh","linear"],
        "use_bias": False,
        "bias_initializer":"zeros",
        "pre_output_dense_units": [8,16]
	}
    bcc = {
        "downsampling_factors": [1,2,3,4],
        "upsampling_factors": [1,2,3,4],
        "filters": 8,
        "conv_kernel_sizes": [13,13,13,13],
        "n_convs": [2,2,2,2],
        "conv_padding_mode": "CONSTANT",
        "conv_conv_activation": tf.nn.leaky_relu,
        "conv_dense_activation": ["tanh","tanh","linear"],
        "conv_pre_output_dense_units": [8,16],
        "conv_use_bias": False,
        "use_resnet": True,
        "conv_downsampling_kernel_sizes": [3,2,3,4]
	}
    fcc = {
        "filters": [8,6,4,3,2,1],
        "kernel_sizes": [11,7,5,5,3,3],
        "padding_mode": "CONSTANT",
        "conv_activation": tf.nn.tanh,
        "dense_activations": ["tanh","tanh","linear"],
        "use_bias": False,
        "pre_output_dense_units": [8,16],
        "bias_initializer":"zeros",
        "final_regular_conv_stages": 4
        }

        
    
    mod = Homogeneous_Poisson_NN_Metalearning(2, use_batchnorm = True, input_normalization = input_norm, output_scaling = output_scaling, pre_bottleneck_convolutions_config = pbcc, bottleneck_config = bcc, final_convolutions_config = fcc, bottleneck_upsampling = 'multilinear')
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
        

        

        


        
