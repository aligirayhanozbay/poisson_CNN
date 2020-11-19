import tensorflow as tf
import math, copy

from .Homogeneous_Poisson_NN_Metalearning import get_init_arguments_from_config, process_normalizations, process_output_scaling_modes, process_regularizer_initializer_and_constraint_arguments
from ..blocks import bottleneck_block_multilinearupsample, bottleneck_block_deconvupsample, resnet
from ..layers import MergeWithAttention, Upsample, JacobiIterationLayer, Scaling
from ..utils import choose_conv_layer, check_batchnorm_fused_enable, apply_advanced_padding_and_call_conv_layer
from ..dataset.utils import compute_domain_sizes, split_indices

class Homogeneous_Poisson_NN_Legacy(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', final_convolutions_config = None, pre_bottleneck_convolutions_config = None, bottleneck_deconv_config = None, bottleneck_multilinear_config = None, input_normalization = None, output_scaling = None, use_batchnorm = False, postsmoother_iterations = 5, use_scaling = False, use_positional_embeddings = True, scaling_config = None, gradient_accumulation_steps = None, bc_type = 'dirichlet'):

        super().__init__()
        ndims = 2
        self.ndims = ndims

        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.input_normalization = process_normalizations(input_normalization)
        self._input_normalization_has_to_be_performed = tf.reduce_any([bool(x) for x in self.input_normalization.values()])
        self.output_scaling = process_output_scaling_modes(output_scaling)
        #self._output_scaling_has_to_be_performed = tf.reduce_any(list(self.output_scaling.values()))
        if self.output_scaling['match_peak_laplacian_magnitude_to_peak_rhs']:
            self.fd_stencil = tf.constant(poisson_CNN.dataset.utils.build_fd_coefficients(5, 2, ndims), dtype=tf.keras.backend.floatx())
            self.fd_conv_method = choose_conv_method(ndims)
        
        self.use_batchnorm = use_batchnorm
        self.use_positional_embeddings = use_positional_embeddings
        self.data_format = data_format

        if pre_bottleneck_convolutions_config is None:
            raise(ValueError('Provide a config for pre bottleneck convolutions'))
        if (bottleneck_deconv_config is None) or (bottleneck_multilinear_config is None):
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
        assert bottleneck_deconv_config['filters'] == bottleneck_multilinear_config['filters']
        fields_in_bottleneck_cfg = ['downsampling_factors', 'upsampling_factors', 'conv_kernel_sizes', 'deconv_kernel_sizes', 'n_convs'] + (['conv_downsampling_kernel_sizes'] if ('conv_downsampling_kernel_sizes' in bottleneck_deconv_config.keys()) else [])
        fields_in_bottleneck_args = ['downsampling_factor', 'upsampling_factor', 'conv_kernel_size', 'deconv_kernel_size', 'n_convs'] + (['conv_downsampling_kernel_size'] if ('conv_downsampling_kernel_sizes' in bottleneck_deconv_config.keys()) else [])
        self.bottleneck_deconv_blocks = [bottleneck_block_deconvupsample(ndims = ndims, data_format = data_format, use_batchnorm = self.use_batchnorm, **get_init_arguments_from_config(bottleneck_deconv_config,k,fields_in_bottleneck_cfg,fields_in_bottleneck_args)) for k in range(len(bottleneck_deconv_config['downsampling_factors']))]
        self.bottleneck_deconv_blocks = sorted(self.bottleneck_deconv_blocks, key = lambda x: x.downsampling_factor, reverse=True)
        
        fields_in_bottleneck_cfg = ['downsampling_factors', 'upsampling_factors', 'conv_kernel_sizes', 'n_convs'] + (['conv_downsampling_kernel_sizes'] if ('conv_downsampling_kernel_sizes' in bottleneck_multilinear_config.keys()) else []) + (['resize_methods'] if ('resize_methods' in bottleneck_multilinear_config.keys()) else [])
        fields_in_bottleneck_args = ['downsampling_factor', 'upsampling_factor', 'conv_kernel_size', 'n_convs'] + (['conv_downsampling_kernel_size'] if ('conv_downsampling_kernel_sizes' in bottleneck_multilinear_config.keys()) else []) + (['resize_method'] if ('resize_methods' in bottleneck_multilinear_config.keys()) else [])
        self.bottleneck_multilinear_blocks = [bottleneck_block_multilinearupsample(ndims = ndims, data_format = data_format, use_batchnorm = self.use_batchnorm, **get_init_arguments_from_config(bottleneck_multilinear_config,k,fields_in_bottleneck_cfg,fields_in_bottleneck_args)) for k in range(len(bottleneck_multilinear_config['downsampling_factors']))]
        self.bottleneck_multilinear_blocks = sorted(self.bottleneck_multilinear_blocks, key = lambda x: x.downsampling_factor, reverse=True)

        self.non_bottleneck_conv = tf.keras.layers.Conv2D(filters = bottleneck_deconv_config['filters'], kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = self.data_format)
    
        #merge bottleneck blocks
        #self.merge = MergeWithAttention(data_format = self.data_format)
        self.post_merge_conv = tf.keras.layers.Conv2D(filters = bottleneck_deconv_config['filters'], kernel_size = 7, activation = tf.nn.leaky_relu, padding = 'same', data_format = self.data_format)
        self.post_merge_resnet = resnet(2, filters = bottleneck_deconv_config['filters'], kernel_size = 7, activation = tf.nn.leaky_relu, data_format = self.data_format)

        #final convolutions
        self.final_convolutions = []
        self.final_convolution_ops = []
        final_convolutions_config = copy.deepcopy(final_convolutions_config)
        final_convolution_stages = len(final_convolutions_config['filters'])
        final_convolutions_padding_mode = final_convolutions_config.pop('padding_mode','CONSTANT')
        final_convolutions_constant_padding_value = final_convolutions_config.pop('constant_padding_value',0.0)
        self.final_regular_conv_stages = final_convolutions_config.pop('final_regular_conv_stages',2)
        for k in range(final_convolution_stages-self.final_regular_conv_stages):
            conv_args = get_init_arguments_from_config(final_convolutions_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv_to_adjust_channel_number = conv_layer(data_format = data_format, padding = 'valid', **conv_args)
            conv = resnet(ndims = 2, use_batchnorm = False, padding_mode = 'constant', data_format = self.data_format, **conv_args)
            self.final_convolutions.append(conv_to_adjust_channel_number)
            self.final_convolution_ops.append(apply_advanced_padding_and_call_conv_layer(final_convolutions_padding_mode, self.final_convolutions[-1], constant_padding_value = final_convolutions_constant_padding_value))
            self.final_convolutions.append(conv)
            self.final_convolution_ops.append(self.final_convolutions[-1])
        for k in range(final_convolution_stages-self.final_regular_conv_stages,final_convolution_stages):
            self.final_convolutions.append(conv_layer(filters = final_convolutions_config['filters'][k], kernel_size = final_convolutions_config['kernel_sizes'][k],activation = tf.keras.activations.linear, padding='same',data_format=self.data_format,use_bias=final_convolutions_config['use_bias']))
            self.final_convolution_ops.append(self.final_convolutions[-1])

        #dx info
        dx_dense_layer_units = [100,100,bottleneck_deconv_config['filters']]
        #dx_dense_layer_units = [100,100,self.final_convolutions[-2].filters]
        dx_dense_layer_activations = [tf.nn.leaky_relu for _ in range(len(dx_dense_layer_units)-1)] + ['linear']
        self.dx_dense_layers = [tf.keras.layers.Dense(dx_dense_layer_units[k], activation = dx_dense_layer_activations[k]) for k in range(len(dx_dense_layer_units))]

        self.postsmoother = JacobiIterationLayer([3,3],[2,2],self.ndims, data_format = self.data_format, n_iterations = postsmoother_iterations) if postsmoother_iterations > 0 else None

        if bc_type.lower() == 'dirichlet':
            self.padding_mode = "CONSTANT"
        elif bc_type.lower() == 'neumann':
            self.padding_mode = "SYMMETRIC"
        else:
            raise(ValueError('bc_type can only be neumann or dirichlet.'))
        self.bc_slice = [Ellipsis] + [slice(1,-1) for _ in range(self.ndims)] + [slice(0,None)] if self.data_format == 'channels_last' else [Ellipsis,slice(0,None)] + [slice(1,-1) for _ in range(self.ndims)]
        self.bc_paddings = [[0,0],[0,0]] + [[1,1] for _ in range(self.ndims)] if self.data_format == 'channels_first' else [[0,0]] + [[1,1] for _ in range(self.ndims)] + [[0,0]]

        self.scaling = Scaling(ndims = self.ndims, data_format = self.data_format, **scaling_config) if use_scaling else None

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
        linspace_start = tf.cast(0.0,tf.keras.backend.floatx())
        linspace_end = tf.cast(1.0,tf.keras.backend.floatx())
        pi = tf.cast(math.pi,tf.keras.backend.floatx())
        pos_embeddings = tf.stack([tf.broadcast_to(tf.reshape(tf.cos(pi * tf.linspace(linspace_start,linspace_end,domain_shape[k])),[1 for _ in range(k)] + [-1] + [1 for _ in range(self.ndims-k-1)]), domain_shape) for k in range(self.ndims)],0 if self.data_format == 'channels_first' else -1)
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
        domain_sizes = self.compute_domain_sizes(tf.concat([dx,dx],1), domain_shape)
        max_domain_sizes = tf.reduce_max(domain_sizes,1)

        if self.use_positional_embeddings:
            pos_embeddings = self.generate_position_embeddings(batch_size, domain_shape)
            conv_inp = tf.concat([rhses, pos_embeddings], 1 if self.data_format == 'channels_first' else -1)#rhses#
        else:
            conv_inp = rhses
            
        dense_inp = tf.concat([dx,domain_sizes],1)
        

        initial_conv_result = self.pre_bottleneck_convolution_ops[0](conv_inp)
        for layer in self.pre_bottleneck_convolution_ops[1:]:
            initial_conv_result = layer(initial_conv_result)
        #bottleneck_results = self.post_merge_resnet(self.post_merge_conv(initial_conv_result))
        #'''
        # bottleneck_results = self.bottleneck_multilinear_blocks[0]([initial_conv_result, domain_sizes])
        # for block in self.bottleneck_multilinear_blocks[1:]:
        #     bottleneck_results = block([tf.concat([bottleneck_results, initial_conv_result],1 if self.data_format == 'channels_first' else -1), domain_sizes])
        # for block in self.bottleneck_deconv_blocks:
        #     bottleneck_results = block(tf.concat([bottleneck_results, initial_conv_result], 1 if self.data_format == 'channels_first' else -1))
        bottleneck_results = []
        for block_num,block in enumerate(self.bottleneck_deconv_blocks):
            bottleneck_results.append(block(initial_conv_result))
        for block in self.bottleneck_multilinear_blocks:
            bottleneck_results.append(block([initial_conv_result, domain_sizes]))

        #bottleneck_results = self.merge(bottleneck_results)
        bottleneck_results = sum(bottleneck_results)/tf.cast(len(bottleneck_results) * tf.shape(bottleneck_results[0])[1 if self.data_format == 'channels_first' else -1], tf.keras.backend.floatx())
        
        bottleneck_results = self.post_merge_resnet(self.post_merge_conv(tf.concat([self.non_bottleneck_conv(initial_conv_result), bottleneck_results], 1 if self.data_format == 'channels_first' else -1)))
        #'''

        #'''
        dx_info = self.dx_dense_layers[0](dense_inp)
        for layer in self.dx_dense_layers[1:]:
            dx_info = layer(dx_info)
        bottleneck_results = tf.einsum('ijkl,ij->ijkl', bottleneck_results, dx_info)

        out = self.final_convolution_ops[0](bottleneck_results)
        #out = self.final_convolution_ops[0](bottleneck_result)
        for layer in self.final_convolution_ops[1:]:
            out = layer(out)
        '''
        dx_info = self.dx_dense_layers[0](dense_inp)
        for layer in self.dx_dense_layers[1:]:
            dx_info = layer(dx_info)
        out = self.final_convolution_ops[0](bottleneck_results)
        for layer in self.final_convolution_ops[1:-1]:
            out = layer(out)
        out = tf.einsum('ijkl,ij->ijkl', out, dx_info)
        out = self.final_convolution_ops[-1](out)
        '''
        
        if self.scaling is not None:
            out = self.scaling([out, rhses])

        out = tf.pad(out[self.bc_slice], self.bc_paddings, mode=self.padding_mode, constant_values = 0.0)

        if self.postsmoother is not None:
            out = self.postsmoother([out,rhses,dx])
        #out = self.scale_outputs(rhses,out,max_domain_sizes,dx)

        return out

    def train_step(self,data):

        inputs, ground_truth = data

        rhses, dx = inputs
        dx = dx[...,:1]

        if self.gradient_accumulation_steps is None:
            with tf.GradientTape() as tape:
                tape.watch(self.trainable_variables)
                pred = self(inputs)
                loss = self.loss_fn(y_true=ground_truth,y_pred=pred,rhs=rhses,dx=tf.concat([dx,dx],1))
            grads = tape.gradient(loss,self.trainable_variables)
        else:
            batch_size = tf.shape(rhses)[0]
            indices = split_indices(batch_size,self.gradient_accumulation_steps)
            grads = None
            for grad_acc_step in range(self.gradient_accumulation_steps):
                with tf.GradientTape() as tape:
                    tape.watch(self.trainable_variables)
                    step_start_idx = indices[grad_acc_step]
                    step_end_idx = indices[grad_acc_step+1]
                    step_rhs = rhses[step_start_idx:step_end_idx]
                    step_dx = dx[step_start_idx:step_end_idx]
                    step_ground_truth = ground_truth[step_start_idx:step_end_idx]
                    pred = self([step_rhs,step_dx])
                    loss = self.loss_fn(y_true=step_ground_truth, y_pred = pred, rhs = step_rhs, dx = tf.concat([step_dx,step_dx],1))
                grads = tape.gradient(loss,self.trainable_variables) if grads is None else [current_grads+new_grads for current_grads,new_grads in zip(grads,tape.gradient(loss,self.trainable_variables))]
            grads = [g/self.gradient_accumulation_steps for g in grads]
        
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        return {'loss' : loss, 'mse': tf.reduce_mean((pred - ground_truth)**2), 'lr': self.optimizer.learning_rate}
    
    def compile(self, loss, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss
'''
class Homogeneous_Poisson_NN_Legacy(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', final_convolutions_config = None, pre_bottleneck_convolutions_config = None, bottleneck_deconv_config = None, bottleneck_multilinear_config = None):
        super().__init__()
        self.data_format = data_format

        if pre_bottleneck_convolutions_config is None:
            raise(ValueError('Provide a config for pre bottleneck convolutions'))
        if (bottleneck_deconv_config is None) or (bottleneck_multilinear_config is None):
            raise(ValueError('Provide a config for bottleneck blocks'))
        if final_convolutions_config is None:
            raise(ValueError('Provide a config for final convolutions'))

        ndims = 2
        conv_layer = choose_conv_layer(ndims)

        self.use_batchnorm = False
        
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
        assert bottleneck_deconv_config['filters'] == bottleneck_multilinear_config['filters']
        fields_in_bottleneck_cfg = ['downsampling_factors', 'upsampling_factors', 'conv_kernel_sizes', 'deconv_kernel_sizes', 'n_convs'] + (['conv_downsampling_kernel_sizes'] if ('conv_downsampling_kernel_sizes' in bottleneck_deconv_config.keys()) else [])
        fields_in_bottleneck_args = ['downsampling_factor', 'upsampling_factor', 'conv_kernel_size', 'deconv_kernel_size', 'n_convs'] + (['conv_downsampling_kernel_size'] if ('conv_downsampling_kernel_sizes' in bottleneck_deconv_config.keys()) else [])
        self.bottleneck_blocks = [bottleneck_block_deconvupsample(ndims = ndims, data_format = data_format, use_batchnorm = self.use_batchnorm, **get_init_arguments_from_config(bottleneck_deconv_config,k,fields_in_bottleneck_cfg,fields_in_bottleneck_args)) for k in range(len(bottleneck_deconv_config['downsampling_factors']))]
        self.n_deconvupsample_bottleneck_blocks = len(self.bottleneck_blocks)
        
        fields_in_bottleneck_cfg = ['downsampling_factors', 'upsampling_factors', 'conv_kernel_sizes', 'n_convs'] + (['conv_downsampling_kernel_sizes'] if ('conv_downsampling_kernel_sizes' in bottleneck_deconv_config.keys()) else [])
        fields_in_bottleneck_args = ['downsampling_factor', 'upsampling_factor', 'conv_kernel_size', 'n_convs'] + (['conv_downsampling_kernel_size'] if ('conv_downsampling_kernel_sizes' in bottleneck_deconv_config.keys()) else [])
        self.bottleneck_blocks = self.bottleneck_blocks + [bottleneck_block_multilinearupsample(ndims = ndims, data_format = data_format, use_batchnorm = self.use_batchnorm, **get_init_arguments_from_config(bottleneck_multilinear_config,k,fields_in_bottleneck_cfg,fields_in_bottleneck_args)) for k in range(len(bottleneck_multilinear_config['downsampling_factors']))]
        self.n_multilinearupsample_bottleneck_blocks = len(self.bottleneck_blocks) 
        
        #merge bottleneck blocks
        self.merge = MergeWithAttention(data_format = self.data_format, n_channels = bottleneck_deconv_config['filters'], n_inputs = self.n_deconvupsample_bottleneck_blocks + self.n_multilinearupsample_bottleneck_blocks)
        self.merge_einsum_str = self.merge.einsum_str.replace('n','')
        self.post_merge_conv = tf.keras.layers.Conv2D(filters = bottleneck_deconv_config['filters'], kernel_size = 7, activation = tf.nn.leaky_relu, padding = 'same')
        self.post_merge_resnet = resnet(2, filters = bottleneck_deconv_config['filters'], kernel_size = 7, activation = tf.nn.leaky_relu)
        
        #dx info
        dx_dense_layer_units = [100,100,bottleneck_deconv_config['filters']]
        dx_dense_layer_activations = [tf.nn.leaky_relu for _ in range(len(dx_dense_layer_units)-1)] + ['linear']
        self.dx_dense_layers = [tf.keras.layers.Dense(dx_dense_layer_units[k], activation = dx_dense_layer_activations[k]) for k in range(len(dx_dense_layer_units))]

        #final convolutions
        self.final_convolutions = []
        self.final_convolution_ops = []
        final_convolutions_config = copy.deepcopy(final_convolutions_config)
        final_convolution_stages = len(final_convolutions_config['filters'])
        final_convolutions_padding_mode = final_convolutions_config.pop('padding_mode','CONSTANT')
        final_convolutions_constant_padding_value = final_convolutions_config.pop('constant_padding_value',0.0)
        self.final_regular_conv_stages = final_convolutions_config.pop('final_regular_conv_stages',2)
        for k in range(final_convolution_stages-self.final_regular_conv_stages):
            conv_args = get_init_arguments_from_config(final_convolutions_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv_to_adjust_channel_number = conv_layer(data_format = data_format, padding = 'valid', **conv_args)
            conv = resnet(ndims = 2, use_batchnorm = False, padding_mode = 'constant', **conv_args)
            self.final_convolutions.append(conv_to_adjust_channel_number)
            self.final_convolution_ops.append(apply_advanced_padding_and_call_conv_layer(final_convolutions_padding_mode, self.final_convolutions[-1], constant_padding_value = final_convolutions_constant_padding_value))
            self.final_convolutions.append(conv)
            self.final_convolution_ops.append(self.final_convolutions[-1])
        for k in range(final_convolution_stages-self.final_regular_conv_stages,final_convolution_stages):
            self.final_convolutions.append(conv_layer(filters = final_convolutions_config['filters'][k], kernel_size = final_convolutions_config['kernel_sizes'][k],activation = tf.keras.activations.linear, padding='same',data_format=self.data_format,use_bias=final_convolutions_config['use_bias']))
            self.final_convolution_ops.append(self.final_convolutions[-1])

    @tf.function
    def call(self, inp):

        times = []
        t0 = time.time()
        rhses, dx = inp
        
        inp_shape = tf.shape(rhses)
        if self.data_format == 'channels_first':
            domain_shape = inp_shape[2:]
        else:
            domain_shape = inp_shape[1:-1]
        batch_size = inp_shape[0]
        domain_sizes = compute_domain_sizes(tf.concat([dx,dx],1), domain_shape)
        max_domain_sizes = tf.reduce_max(domain_sizes,1)
        pos_embeddings = generate_position_embeddings(batch_size, domain_shape)

        dense_inp = tf.concat([dx,domain_sizes],1)
        conv_inp = tf.concat([rhses, pos_embeddings], 1 if self.data_format == 'channels_first' else -1)
        times.append(time.time() - t0)
        
        initial_conv_result = self.pre_bottleneck_convolution_ops[0](conv_inp)
        for layer in self.pre_bottleneck_convolution_ops[1:]:
            initial_conv_result = layer(initial_conv_result)
        times.append(time.time() - t0 - sum(times))
            
        out = tf.zeros(initial_conv_result.shape, dtype = initial_conv_result.dtype)
        for block_num in range(self.n_deconvupsample_bottleneck_blocks):
            out += tf.einsum(self.merge_einsum_str, self.bottleneck_blocks[block_num](initial_conv_result), self.merge.attention_weights[block_num])
        for block_num in range(self.n_deconvupsample_bottleneck_blocks, len(self.bottleneck_blocks)):
            out += tf.einsum(self.merge_einsum_str, self.bottleneck_blocks[block_num]([initial_conv_result, domain_sizes]), self.merge.attention_weights[block_num])
        times.append(time.time() - t0 - sum(times))
                 
        # bottleneck_results = []
        # for block in self.bottleneck_blocks:
        #     if type(block) is bottleneck_block_multilinearupsample:
        #         bottleneck_results.append(block([initial_conv_result, domain_sizes]))
        #     else:
        #         bottleneck_results.append(block(initial_conv_result))

        #out = self.merge(bottleneck_results)

        out = self.post_merge_resnet(self.post_merge_conv(out))
        dx_info = self.dx_dense_layers[0](dense_inp)
        for layer in self.dx_dense_layers[1:]:
            dx_info = layer(dx_info)
        out = tf.einsum('ijkl,ij->ijkl', out, dx_info)
        times.append(time.time() - t0 - sum(times))
                     
        for layer in self.final_convolution_ops:
            out = layer(out)
        times.append(time.time() - t0 - sum(times))
        times = tf.stack(times,0)
        return out, times

    def train_step(self,data):

        inputs, ground_truth = data

        rhses, dx = inputs

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            pred = self(inputs)
            
            loss = self.loss_fn(y_true=ground_truth,y_pred=pred,rhs=rhses,dx=dx)
        grads = tape.gradient(loss,self.trainable_variables)
        
        #for k in range(len(grads)):
        #    gradk = tf.debugging.check_numerics(grads[k], 'nan or inf in grad ' + str(k))
        #    #grads[k] = tf.clip_by_norm(grads[k],tf.constant(0.5))
        
        
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        return {'loss' : loss, 'mse': tf.reduce_mean((pred - ground_truth)**2)}# 'peak_rhs' : tf.reduce_max(tf.abs(rhses)), 'peak_soln': tf.reduce_max(tf.abs(ground_truth)), 'peak_pred':tf.reduce_max(tf.abs(pred)), 'model peak log' : model_peak_log, 'grad peak log' : grad_peak_log}

    def compile(self, loss, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss
'''
if __name__ == '__main__':
    import time
    pbcc = {
        "filters": [4,16,32],
        "kernel_sizes": [19,17,15],
        "padding_mode": "symmetric",
        "activation": tf.nn.leaky_relu,
        "use_bias": False,
        "bias_initializer":"zeros"
	}
    bcc_ml = {
        "downsampling_factors": [32,64],
        "upsampling_factors": [32,64],
        "filters": 32,
        "conv_kernel_sizes": [5,5],
        "n_convs": [2,2],
        "padding_mode": "CONSTANT",
        "constant_padding_value": 4.0,
        "conv_activation": tf.nn.leaky_relu,
        "conv_use_bias": False,
        "use_resnet": True,
        "downsampling_method": "pool",
        "pool_downsampling_method": "average"
	}
    bcc_d = {
        "downsampling_factors": [1,2,4,8,16],
        "upsampling_factors": [1,2,4,8,16],
        "filters": 32,
        "conv_kernel_sizes": [13,13,13,13,13],
        "deconv_kernel_sizes": [13,2,4,8,16],
        "n_convs": [2,2,2,2,2],
        "padding_mode": "CONSTANT",
        "conv_activation": tf.nn.leaky_relu,
        "conv_use_bias": True,
        "use_resnet": True,
        "downsampling_method": "pool",
        "pool_downsampling_method": "average"
    }
    # bcc_d = {
    #     "downsampling_factors": [1,2,4,8,16,32,64],
    #     "upsampling_factors": [1,2,4,8,16,32,64],
    #     "filters": 32,
    #     "conv_kernel_sizes": [13,13,13,13,13,5,5],
    #     "deconv_kernel_sizes": [13,2,4,8,16,32,64],
    #     "n_convs": [2,2,2,2,2,2,2],
    #     "padding_mode": "CONSTANT",
    #     "conv_activation": tf.nn.leaky_relu,
    #     "conv_use_bias": True,
    #     "use_resnet": True,
    #     "downsampling_method": "pool",
    #     "pool_downsampling_method": "average"
    # }
    fcc = {
        "filters": [8,6,4,3,2,1],
        "kernel_sizes": [11,7,5,5,3,3],
        "padding_mode": "CONSTANT",
        "constant_padding_value": 2.0,
        "activation": tf.nn.tanh,
        "use_bias": False,
        "bias_initializer":"zeros",
        "final_regular_conv_stages": 2
        }
    mod = Homogeneous_Poisson_NN_Legacy(final_convolutions_config = fcc, pre_bottleneck_convolutions_config = pbcc, bottleneck_deconv_config = bcc_d, bottleneck_multilinear_config = bcc_ml)

    '''
    pbcc = {
        "filters": [4,8,16],
        "kernel_sizes": [19,17,15],
        "padding_mode": "symmetric",
        "activation": tf.nn.leaky_relu,
        "use_bias": False,
        "bias_initializer":"zeros"
	}
    bcc = {
        "downsampling_factors": [1,2,3,4,8,16,32,64],
        "upsampling_factors": [1,2,3,4,8,16,32,64],
        "filters": 16,
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
    
    mod = Homogeneous_Poisson_NN_Legacy(use_batchnorm = False, pre_bottleneck_convolutions_config = pbcc, bottleneck_config = bcc, final_convolutions_config = fcc, bottleneck_upsampling = 'multilinear')
    '''
    bsize = 1
    rhs = 2*tf.random.uniform((bsize,1,1000,1000))-1
    dx = tf.random.uniform((bsize,1))
    mod([rhs,dx])
    mod.summary()
    ntrials = 20
    t0 = time.time()
    for k in range(ntrials):
        t00 = time.time()
        s = mod([rhs,dx])
        t01 = time.time()
        print(t00 - t01)
        print(s.shape)
        #print(tf.reduce_sum(t))
    t1 = time.time()
    print(sum(times)/len(times))
    print(s.shape)
