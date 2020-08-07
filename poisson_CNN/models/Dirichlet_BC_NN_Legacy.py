import copy, math, warnings
import tensorflow as tf

from .Homogeneous_Poisson_NN_Metalearning import get_init_arguments_from_config, process_normalizations, process_output_scaling_modes, process_regularizer_initializer_and_constraint_arguments
from ..layers import Upsample, SpatialPyramidPool, JacobiIterationLayer
from ..blocks import resnet
from ..utils import check_batchnorm_fused_enable, apply_advanced_padding_and_call_conv_layer, choose_conv_layer
from ..dataset.utils import set_max_magnitude_in_batch, compute_domain_sizes
'''
class Dirichlet_BC_NN_Legacy(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', conv1d_config = None, dx_config = None, conv2d_config = None):
        pass
'''
class Dirichlet_BC_NN_Legacy_2(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', boundary_conv_config = None, spp_config = None, domain_info_mlp_config = None, final_convolutions_config = None, postsmoother_iterations = 0, use_batchnorm = False):

        super().__init__()
        ndims = 2
        self.ndims = ndims#ndims for the output
        self.data_format = data_format

        # self.input_normalization = {'rhs_max_magnitude': True}
        self.use_batchnorm = use_batchnorm

        if boundary_conv_config is None:
            raise(ValueError('Provide a config for the boundary convolutions.'))
        if spp_config is None:
            raise(ValueError('Provide a config for the Spatial Pyramid Pooling.'))
        if final_convolutions_config is None:
            raise(ValueError('Provide a config for the domain convolutions.'))
        if domain_info_mlp_config is None:
            raise(ValueError('Provide a config for the domain info MLP.'))

        boundary_conv_layer = choose_conv_layer(self.ndims-1)
        domain_conv_layer = choose_conv_layer(self.ndims)

        #last number of filters for the BC convs and the MLP layers must be identical since this is the # of modes used for the x-direction sinh basis functions
        assert boundary_conv_config['filters'][-1] == domain_info_mlp_config['units'][-1]
        if boundary_conv_config['filters'][-1] > 27 and (tf.keras.backend.floatx() == 'float32'):
            warnings.warn(str(boundary_conv_config['filters'][-1]) + ' sinh modes chosen may lead to NaN values with float32 precision. Consider using fewer than 28 when using float32.')
        
        self.x_dir_nmodes = domain_info_mlp_config['units'][-1]

        #convolutions on the BC info
        fields_in_conv_cfg = ['filters','kernel_sizes']
        fields_in_conv_args = ['filters','kernel_size']
        self.boundary_convolutions = []
        self.boundary_convolution_ops = []
        boundary_conv_config = copy.deepcopy(boundary_conv_config)
        boundary_conv_padding_mode = boundary_conv_config.pop('padding_mode','CONSTANT')
        boundary_conv_constant_padding_value = boundary_conv_config.pop('constant_padding_value',0.0)
        for k in range(len(boundary_conv_config['filters'])):
            conv_args = get_init_arguments_from_config(boundary_conv_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv = boundary_conv_layer(data_format = self.data_format, padding = 'valid', **conv_args)
            self.boundary_convolutions.append(conv)
            self.boundary_convolution_ops.append(apply_advanced_padding_and_call_conv_layer(boundary_conv_padding_mode, self.boundary_convolutions[-1], constant_padding_value = boundary_conv_constant_padding_value))
            if self.use_batchnorm:
                batchnorm_fused_enable = check_batchnorm_fused_enable(ndims = self.ndims-1)
                bnorm = tf.keras.layers.BatchNormalization(axis = 1 if self.data_format == 'channels_first' else -1, fused = batchnorm_fused_enable)
                self.boundary_convolutions.append(bnorm)
                self.boundary_convolution_ops.append(bnorm)
            resnet_block = resnet(self.ndims-1, use_batchnorm = self.use_batchnorm, padding_mode = boundary_conv_padding_mode, constant_padding_value = boundary_conv_constant_padding_value, data_format = self.data_format, **conv_args)
            self.boundary_convolutions.append(resnet_block)
            self.boundary_convolution_ops.append(resnet_block)
        self._einsum_boundary_conv_input_str = 'bm...' if self.data_format == 'channels_first' else 'b...m'
        self._einsum_output_str = 'bmx...' if self.data_format == 'channels_first' else 'bx...m'

        #dense layers for the domain info + SPP result from boundary convs
        self.spp = SpatialPyramidPool(ndims = self.ndims-1, data_format = self.data_format, receive_padded_values = False, **spp_config)
        fields_in_dense_cfg = ['units', 'activations']
        fields_in_dense_args = ['units', 'activation']
        self.domain_info_dense_layers = []
        for k in range(len(domain_info_mlp_config['units'])):
            dense_args = get_init_arguments_from_config(domain_info_mlp_config,k,fields_in_dense_cfg,fields_in_dense_args)
            self.domain_info_dense_layers.append(tf.keras.layers.Dense(**dense_args))

        #convolutions on the assembled (ndims+2)-dimensional tensor
        self.final_convolutions = []
        self.final_convolution_ops = []
        final_convolutions_config = copy.deepcopy(final_convolutions_config)
        final_convolution_stages = len(final_convolutions_config['filters'])
        final_convolutions_padding_mode = final_convolutions_config.pop('padding_mode','CONSTANT')
        final_convolutions_constant_padding_value = final_convolutions_config.pop('constant_padding_value',0.0)
        self.final_regular_conv_stages = final_convolutions_config.pop('final_regular_conv_stages',2)
        for k in range(final_convolution_stages-self.final_regular_conv_stages):
            conv_args = get_init_arguments_from_config(final_convolutions_config,k,fields_in_conv_cfg,fields_in_conv_args)
            conv_to_adjust_channel_number = domain_conv_layer(data_format = data_format, padding = 'valid', **conv_args)
            conv = resnet(ndims = self.ndims, use_batchnorm = False, padding_mode = 'constant', data_format = self.data_format, **conv_args)
            self.final_convolutions.append(conv_to_adjust_channel_number)
            self.final_convolution_ops.append(apply_advanced_padding_and_call_conv_layer(final_convolutions_padding_mode, self.final_convolutions[-1], constant_padding_value = final_convolutions_constant_padding_value))
            self.final_convolutions.append(conv)
            self.final_convolution_ops.append(self.final_convolutions[-1])
        for k in range(final_convolution_stages-self.final_regular_conv_stages,final_convolution_stages):
            self.final_convolutions.append(domain_conv_layer(filters = final_convolutions_config['filters'][k], kernel_size = final_convolutions_config['kernel_sizes'][k],activation = tf.nn.tanh, padding='same',data_format=self.data_format,use_bias=final_convolutions_config['use_bias']))
            self.final_convolution_ops.append(self.final_convolutions[-1])

        self.postsmoother = JacobiIterationLayer([3,3],[2,2],self.ndims, data_format = self.data_format, n_iterations = postsmoother_iterations) if postsmoother_iterations > 0 else None

        self.direction_perpendicular_to_bc = 2 if self.data_format == 'channels_first' else 1
        self.concat_bc_output_slice = tuple([Ellipsis, slice(1,None)] + [slice(0,None) for _ in range(self.ndims-1)])

    @tf.function
    def get_domain_shape(self,inpshape):
        return (inpshape[2:] if self.data_format == 'channels_first' else inpshape[1:-1])

    @tf.function
    def build_series_x_dir_components(self,x_output_resolution):
        xbar = tf.linspace(tf.constant(0.0,tf.keras.backend.floatx()), tf.constant(1.0,tf.keras.backend.floatx()), x_output_resolution)
        mode_numbers = tf.range(1,self.x_dir_nmodes+1,dtype=xbar.dtype)
        sinh_arguments = tf.einsum('m,x->mx',mode_numbers,math.pi*(xbar-1))
        sinh_values = set_max_magnitude_in_batch(tf.math.sinh(sinh_arguments), tf.constant(1.0,dtype=xbar.dtype))
        return sinh_values

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
    def call(self,inp):
        bc, dx, x_output_resolution = inp

        bc_shape = tf.shape(bc)
        domain_shape = tf.concat([tf.expand_dims(x_output_resolution,0),self.get_domain_shape(bc_shape)],0)
        domain_sizes = compute_domain_sizes(tf.concat([dx,dx],1), domain_shape)
        max_domain_sizes = tf.reduce_max(domain_sizes,1,keepdims=True)
        pos_embeddings_nd = self.generate_position_embeddings(bc_shape[0],domain_shape)
        pos_embeddings_nminus1d = pos_embeddings_nd[...,0,:]

        #(ndims-1) dimensional convolutions on the BC info
        bc_conv_result = self.boundary_convolution_ops[0](tf.concat([bc,pos_embeddings_nminus1d], 1 if self.data_format == 'channels_first' else -1))
        for layer in self.boundary_convolution_ops[1:]:
            bc_conv_result = layer(bc_conv_result)

        #mlp layers
        bc_conv_spp_result = self.spp(bc_conv_result)
        dense_inp = tf.concat([dx,domain_sizes/max_domain_sizes,bc_conv_spp_result],1)
        dense_result = self.domain_info_dense_layers[0](dense_inp)
        for layer in self.domain_info_dense_layers[1:]:
            dense_result = layer(dense_result)

        #sinh values as the x direction component of the solution
        sinh_values = self.build_series_x_dir_components(x_output_resolution)
        
        #einsum the three components together
        out = tf.einsum(self._einsum_boundary_conv_input_str + ',mx,bm->' + self._einsum_output_str, bc_conv_result, sinh_values, dense_result)

        #apply 2d convs
        out = tf.concat([out,pos_embeddings_nd], 1 if self.data_format == 'channels_first' else -1)
        for layer in self.final_convolution_ops:
            out = layer(out)
	
        out = set_max_magnitude_in_batch(out, tf.constant(1.0,tf.keras.backend.floatx()))
        out = tf.reshape(out, tf.stack([bc_shape[0],bc_shape[1],x_output_resolution,bc_shape[2]]))
        out = tf.concat([tf.expand_dims(bc,self.direction_perpendicular_to_bc), out[self.concat_bc_output_slice]], self.direction_perpendicular_to_bc)

        #apply post smoothing if desired
        if self.postsmoother is not None:
            out = self.postsmoother([out,tf.zeros(out.shape,dtype=out.dtype),dx])

        return out

    def train_step(self,data):

        inputs, ground_truth = data
        bc,dx = inputs
        x_output_resolution = tf.shape(ground_truth)[2 if self.data_format == 'channels_first' else 1]

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            pred = self([bc,dx,x_output_resolution])
            loss = self.loss_fn(y_true = ground_truth, y_pred = pred, rhs = tf.zeros(tf.shape(ground_truth),dtype=ground_truth.dtype), dx = tf.concat([dx,dx],1))
        grads = tape.gradient(loss,self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        return {'loss': loss, 'mse': tf.reduce_mean((pred - ground_truth)**2), 'lr': self.optimizer.learning_rate}

    def compile(self, loss, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss

        

if __name__ == '__main__':
    bsize = 10
    nx = 101
    ny = 75
    dx = tf.random.uniform((bsize,1))
    bc = tf.random.uniform((bsize,1,ny))
    x_output_resolution = tf.constant(nx,dtype=tf.int32)

    bccfg = {
        'filters': [4,8,16,32,20],
        'kernel_sizes': [19,17,15,13,11],
        'padding_mode': 'SYMMETRIC',
        'activation': tf.nn.leaky_relu,
        'use_bias': True
    }
    sppcfg = {
        'levels': [[2],3,5,8],
        'pooling_type': 'average'
    }
    mlpcfg = {
        'units': [250,125,20],
        'activations': [tf.nn.leaky_relu,tf.nn.leaky_relu,'softmax']
    }
    fccfg = {
        'filters': [32,16,8,4,2,1],
        'kernel_sizes': [7,5,3,3,3,3],
        'padding_mode': 'CONSTANT',
        'constant_padding_value': 0.0,
        'final_regular_conv_stages': 3,
        'use_bias': True
    }

    mod = Dirichlet_BC_NN_Legacy_2(data_format = 'channels_first', boundary_conv_config = bccfg, spp_config = sppcfg, domain_info_mlp_config = mlpcfg, final_convolutions_config = fccfg, postsmoother_iterations = 0, use_batchnorm = True)
    out = mod([bc,dx,x_output_resolution])

    import pdb
    pdb.set_trace()
        

        
           
        

        
        
