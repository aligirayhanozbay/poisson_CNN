import tensorflow as tf
import numpy as np
import itertools
import opt_einsum as oe
import string

from .custom_blocks import ResampledConvolutionBlock, ResnetBlock, AveragePoolingBlock, Scaling
from .Model_With_Integral_Loss import Model_With_Integral_Loss_ABC
from ...layers import MergeWithAttention
from ...layers import Upsample
from ...layers import SpatialPyramidPool
#from ..layers import Scaling

def channels_first_rot_90(image,k=1):
    if len(image.shape) == 4:
        image = tf.transpose(image, (0,2,3,1))
    elif len(image.shape) == 3:
        image = tf.transpose(image, (0,2,1))
        image_was_rank3 = True
    elif len(image.shape) == 2:
        image = tf.expand_dims(image, axis = 2)
        image_was_rank3 = False
    else:
        raise ValueError('image must be a rank 2,3 or 4 Tensor')
    
    image = tf.image.rot90(image, k = k)
    
    if len(image.shape) == 4:
        return tf.transpose(image, (0,3,1,2))
    elif image_was_rank3:
        return tf.transpose(image, (0,2,1))
    else:
        return image[...,0]
    
class Homogeneous_Poisson_NN_Fluidnet(Model_With_Integral_Loss_ABC):
    '''
    Takes a tf.Tensor containing the RHS and grid spacing information, and gives the corresponding solution to the Poisson problem with homogeneous BCs.
    '''
    def __init__(self, pooling_block_number = 6, post_dx_einsum_conv_block_number = 5, initial_kernel_size = 19, final_kernel_size = 3, resize_methods = None, data_format = 'channels_first', use_batchnorm = False, use_deconv_upsample = False, kernel_regularizer = None, bias_regularizer = None, use_scaling = True, scaling_args = {'downsampling_ratio_per_stage': 3, 'stages': 3, 'filters': 4, 'activation': tf.nn.leaky_relu, 'spp_levels': [[2,2],3,5]}, **kwargs):
        '''
        Init arguments:

        data_format: same as keras
        pooling_block_number: integer. controls the number of simultaneous pooling 'threads'. each pooling thread will apply average pooling with pool_size 2^k (k = [1,...,pooling_block_number]), do convoltuions and then upsample to original resolution. choose such that 2^(pooling_block_number) is close as possible to the minimum grid size your inputs will have.
        use_deconv_upsample: boolean. if set to true, pooling blocks will use transposed convolution to upsample. last 2 pooling layers will use bilinear and nearest neighbor unless overridden by resize_methods
        resize_methods: list of tf.image.ResizeMethod members. used to choose resizing algorithms for pooling block upsampling.
        use_batchnorm: boolean. determines if batch norm layers should be used. if set to true, supply __call__ argument training = False when performing inference.
        post_dx_einsum_conv_block_number: integer. no of convolution blocks after the result of the domain info processing is merged.
        initial_kernel_size: integer. initial kernel size of convolutions. most convolutions will have this kernel size.
        final_kernel_size: integer. the kernel size of the final convolutions will progressively move towards this number. keep it small to prevent output artefacting near image edges.
        kernel_regularizer: same as the corresponding tf.keras.layers.Conv2D argument
        bias_regularizer: same as the corresponding tf.keras.layers.Conv2D argument

        **kwargs: used to set self.integral_loss parameters for the Model_With_Integral_Loss_ABC abstract base class
        '''
        super().__init__(**kwargs)
        self.training = True
        self.pooling_block_number = pooling_block_number
        self.data_format = data_format
        self.pooling_block_kernel_sizes = 9 * np.ones((self.pooling_block_number), dtype = np.int32)
        self.pooling_block_kernel_sizes[-2:] = 3
        self.pooling_block_kernel_sizes = list(self.pooling_block_kernel_sizes)
        self.use_batchnorm = use_batchnorm

        final_dense_layer_units = 32
        
        if not resize_methods: #set resizing methods for pooling blocks if using bicubic upsampling
            try:
                self.resize_methods = [tf.compat.v1.image.ResizeMethod.BICUBIC for i in range(self.pooling_block_number-2)] + [tf.compat.v1.image.ResizeMethod.BILINEAR, tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR]
            except:
                self.resize_methods = [tf.image.ResizeMethod.BICUBIC for i in range(self.pooling_block_number-2)] + [tf.image.ResizeMethod.BILINEAR, tf.image.ResizeMethod.NEAREST_NEIGHBOR]
        else:
            self.resize_methods = resize_methods

        self.pooling_block_filters = 32
        
        self.conv_1 = tf.keras.layers.Conv2D(filters = 8, kernel_size = initial_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        #self.batchnorm_1 = tf.keras.layers.BatchNormalization(axis = 1)
        
        self.conv_2 = tf.keras.layers.Conv2D(filters = self.pooling_block_filters, kernel_size = initial_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        #self.batchnorm_2 = tf.keras.layers.BatchNormalization(axis = 1)
        
        self.conv_3 = tf.keras.layers.Conv2D(filters = self.pooling_block_filters, kernel_size = initial_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)

        if use_deconv_upsample:
            use_deconv_upsample = [True for k in range(self.pooling_block_number-2)] + [False for k in range(2)]
        else:
            use_deconv_upsample = [False for k in range(self.pooling_block_number)]
        self.pooling_blocks = [AveragePoolingBlock(2**(i+1), resize_method = self.resize_methods[i], data_format = data_format, kernel_size = int(self.pooling_block_kernel_sizes[i]), filters = self.pooling_block_filters, activation = tf.nn.leaky_relu, use_resnetblocks = True, use_deconv_upsample = use_deconv_upsample[i], kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer) for i in range(pooling_block_number)]
        
        self.merge = MergeWithAttention()
        
        self.dx_einsum_conv = tf.keras.layers.Conv2D(filters = final_dense_layer_units, kernel_size = initial_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        self.dx_einsum_resnet = ResnetBlock(filters = final_dense_layer_units, kernel_size = initial_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)

        self.post_dx_einsum_conv_blocks = []
        for k in range(post_dx_einsum_conv_block_number):
            ksize = initial_kernel_size + (k*(final_kernel_size - initial_kernel_size)//(post_dx_einsum_conv_block_number-1))
            filters = final_dense_layer_units-((k*final_dense_layer_units)//(post_dx_einsum_conv_block_number))
            block = []
            block.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = ksize, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            block.append(ResnetBlock(filters = filters, kernel_size = ksize, activation=tf.nn.leaky_relu, data_format=data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            self.post_dx_einsum_conv_blocks.append(block)

        if use_batchnorm:
            self.batchnorm_layers = []
            if self.data_format == 'channels_first':
                self.batchnorm_last = tf.keras.layers.BatchNormalization(axis = 1)
            else:
                self.batchnorm_last = tf.keras.layers.BatchNormalization(axis = -1)
            for k in range(post_dx_einsum_conv_block_number):
                if self.data_format == 'channels_first':
                    self.batchnorm_layers.append(tf.keras.layers.BatchNormalization(axis = 1))
                else:
                    self.batchnorm_layers.append(tf.keras.layers.BatchNormalization(axis = -1))

        self.conv_last = tf.keras.layers.Conv2D(filters = 1, kernel_size = final_kernel_size, activation='linear', data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        self.resnet_last = ResnetBlock(filters = 1, kernel_size = final_kernel_size, activation='linear', data_format=data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        
        self.dx_dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dx_dense_1 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dx_dense_2 = tf.keras.layers.Dense(final_dense_layer_units, activation = 'linear')

        self.use_scaling = use_scaling
        if use_scaling:
            self.scaling = Scaling(data_format = self.data_format, kernel_size = final_kernel_size, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, **scaling_args)
        
    def call(self, inp):
        '''
        input arguments:
        
        inp[0]: tf.Tensor of shape (batch_size, 1, nx, ny) or (batch_size, nx, ny, 1) based on self.data_format. this is the RHSes.
        inp[1]: tf.Tensor of shape (batch_size, 1). this must be the grid spacing information.
        '''
        self.dx = inp[1]
        if self.data_format == 'channels_first':#compute domain info
            domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[3]), tf.keras.backend.floatx())], axis = 0))
        else:
            domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[1]), tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx())], axis = 0))

        out = self.conv_1(inp[0]) #initial convolutions
        out = self.conv_2(out)
        
        out = self.merge([self.conv_3(out)] + [pb(out) for pb in self.pooling_blocks])#pooling 'threads'

        out = self.dx_einsum_conv(out)
        out = self.dx_einsum_resnet(out)
        
        out = tf.einsum('ijkl, ij -> ijkl',out, self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(domain_info)))) #incorporate domain info

        for block_num, block in enumerate(self.post_dx_einsum_conv_blocks): #perform additional convolutions
            for layer in block:
                out = layer(out)
            if self.use_batchnorm:
                out = self.batchnorm_layers[block_num](out, training = self.training)

        if self.use_batchnorm:
            out = self.batchnorm_last(out, training = self.training)
        out = self.conv_last(out)
        out = self.resnet_last(out)

        if self.use_scaling:
            out = self.scaling([out, inp[0]])

        if self.training == False:
            out = tf.einsum('i...,i->i...', out, tf.reduce_prod(domain_info[:,1:], axis = 1))
            
        return out

    def __call__(self, inp, training = True):#overload __call__ to allow freezing batch norm parameters
        self.training = training
        return super().__call__(inp)
    

class Homogeneous_Poisson_NN_Fourier(Model_With_Integral_Loss_ABC): #variant to include dx info
    def __init__(self, filters = [8,16,32,64,16,8], kernel_sizes = [5,5,5,5,5,5], pool_after_kth_resnet_block = None, pool_size = 2, nmodes = (32,32), pyramid_pooling_params = {'levels': [[3,3],6,9,12], 'pooling_type': 'AVG'}, data_format = 'channels_first', domain_info_dense_layer_units = [100,100,16], final_dense_layer_units = [250, 500], kernel_regularizer = None, bias_regularizer = None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format
        if self.data_format == 'channels_first':
            self.space_dims_start = 2
        else:
            self.space_dims_start = 1
        self.space_dims_end = self.space_dims_start + len(nmodes)
        self.resnet_blocks = []
        self.nmodes = nmodes
        self.total_mode_number = int(tf.reduce_prod(nmodes))
        
        for i in range(len(kernel_sizes)):
            self.resnet_blocks.append(tf.keras.layers.Conv2D(filters = filters[i], kernel_size = kernel_sizes[i], data_format = self.data_format, activation = tf.nn.leaky_relu, padding = 'same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            self.resnet_blocks.append(ResnetBlock(filters = filters[i], kernel_size = kernel_sizes[i], data_format = self.data_format, activation = tf.nn.leaky_relu, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            if (pool_after_kth_resnet_block is not None) and (i % pool_after_kth_resnet_block == 0):
                self.resnet_blocks.append(tf.keras.layers.MaxPool2D(pool_size = 2, padding = 'same', data_format = self.data_format))
        self.spp = SpatialPyramidPool(data_format = self.data_format, **pyramid_pooling_params)

        self.domain_info_dense_layers = []
        for units in domain_info_dense_layer_units:
            self.domain_info_dense_layers.append(tf.keras.layers.Dense(units, activation = tf.nn.leaky_relu, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))

        self.final_dense_layers = []
        for units in final_dense_layer_units:
            self.final_dense_layers.append(tf.keras.layers.Dense(units, activation = tf.nn.leaky_relu, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
        self.final_dense_layers.append(tf.keras.layers.Dense(tf.reduce_prod(nmodes), activation = lambda x: 4*tf.nn.tanh(x)/np.pi**2))

        self.mplus1_pi_squared = tf.cast(tf.stack(tf.meshgrid(*[((np.arange(mode_number)+1)*np.pi)**2 for mode_number in self.nmodes], indexing = 'ij'), axis = 0), tf.keras.backend.floatx())
        # self.dx_dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        # self.dx_dense_1 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        # self.dx_dense_2 = tf.keras.layers.Dense(16, activation = tf.nn.leaky_relu)
        
        # self.fourier_dense_0 = tf.keras.layers.Dense(250, activation = tf.nn.leaky_relu)
        # self.fourier_dense_1 = tf.keras.layers.Dense(500, activation = tf.nn.leaky_relu)
        # self.fourier_dense_2 = tf.keras.layers.Dense(tf.reduce_prod(nmodes), activation = lambda x: 4*tf.nn.tanh(x)/np.pi**2)

        # self.scaling = Scaling()

        
        # m,n = tf.meshgrid(np.arange(self.nmodes[0])+1,np.arange(self.nmodes[1])+1)
        # self.m = tf.expand_dims((tf.cast(m, tf.keras.backend.floatx())**2) * (tf.cast(np.pi, tf.keras.backend.floatx())**2), axis = 0)
        # self.n = tf.expand_dims((tf.cast(n, tf.keras.backend.floatx())**2) * (tf.cast(np.pi, tf.keras.backend.floatx())**2), axis = 0)

    def call(self,inp):
        inpshape = tf.keras.backend.shape(inp[0])
        self.dx = inp[1]
        domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.keras.backend.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx())] + [tf.cast(inpshape[k], tf.keras.backend.floatx()) for k in range(self.space_dims_start, self.space_dims_end)], axis = 0))
        
        grids = tf.meshgrid(*[tf.linspace(0.0,1.0,tf.keras.backend.cast(inpshape[k], tf.int32)) for k in range(self.space_dims_start, self.space_dims_end)], indexing = 'ij')
        for grid in grids:
            grid = tf.cast(grid, tf.keras.backend.floatx())
            
        sine_values = tf.Variable(lambda: tf.tile(tf.expand_dims(tf.keras.backend.zeros(tf.shape(grids[0]), dtype = tf.keras.backend.floatx()), axis = 0), [self.total_mode_number] + [1 for dim in grids[0].shape]))
        for m in range(self.total_mode_number):
            indices = np.zeros((len(self.nmodes),), dtype = np.int64)
            indices[-1] = m % self.nmodes[-1]
            for k in range(len(self.nmodes)-1):
                indices[k] = int(np.floor(m/np.prod(self.nmodes[k+1:]))) % self.nmodes[k]
            sine_val = tf.Variable(lambda: tf.sin((indices[0]+1)*np.pi*grids[0]))
            for k in range(1, len(indices)):
                sine_val.assign(sine_val * tf.sin((indices[k]+1)*np.pi*grids[k]))
            sine_values[m].assign(sine_val)

        sine_values = tf.keras.backend.reshape(sine_values, tf.concat([list(self.nmodes) , tf.shape(grids[0])], axis = 0))

        dx_res = self.domain_info_dense_layers[0](domain_info)
        for layer in self.domain_info_dense_layers[1:]:
            dx_res = layer(dx_res)

        amplitudes = self.resnet_blocks[0](inp[0])
        for layer in self.resnet_blocks[1:]:
            amplitudes = layer(amplitudes)
        amplitudes = tf.concat([self.spp(amplitudes),dx_res], axis = 1)

        for layer in self.final_dense_layers:
            amplitudes = layer(amplitudes)
            
        amplitudes = tf.keras.backend.reshape(amplitudes, [-1] + list(self.nmodes))
        amplitudes = -amplitudes * (2**(len(inpshape)-2)) / tf.einsum('bi,i...->b...', 1/(domain_info[:,1:])**2,self.mplus1_pi_squared)

        contr_expr = 'z' + string.ascii_lowercase[:len(amplitudes.shape)-1] + ',' + string.ascii_lowercase[:len(amplitudes.shape)-1] + '...->z...'
        out = tf.einsum(contr_expr, amplitudes, sine_values)

        if self.data_format == 'channels_first':
            return tf.expand_dims(out, axis = 1)
        else:
            return tf.expand_dims(out, axis = -1)

################# TEST SECTION #####################
from ..dataset.generators.numerical import set_max_magnitude_in_batch as smmib
class Scaling_Test(tf.keras.models.Model):
    def __init__(self, stages = 2, downsampling_ratio_per_stage = 2, data_format = 'channels_first', padding = 'same', spp_levels = [[2,2], 3, 5], **convargs):
        super().__init__()
        self.spp = SpatialPyramidPool(spp_levels, data_format = data_format, pooling_type = 'MAX')
        self.stages = []
        self.data_format = data_format
        for k in range(stages):
            self.stages.append(tf.keras.layers.Conv2D(data_format = data_format, padding = padding, **convargs))
            self.stages.append(tf.keras.layers.AveragePooling2D(pool_size = downsampling_ratio_per_stage, padding = padding, data_format = data_format))

        self.dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dense_1 = tf.keras.layers.Dense(25, activation = tf.nn.leaky_relu)
        self.dense_2 = tf.keras.layers.Dense(1, activation = 'linear')

    def call(self, inp):
        #input format: tf.Tensor of shape [batch_size, 2, nx, ny] where [:,0,...] is the input to conditionally scale, or a list of 2 tf.Tensors such that [input_to_scale, 2nd input]
        domain_info = inp[2]
        if self.data_format == 'channels_first' and isinstance(inp, list):
            inp = tf.concat(inp[:2], axis = 1)
            inp_to_scale = tf.expand_dims(inp[:,0,...], axis = 1)
        else:
            inp = tf.concat(inp[:2], axis = -1)
            inp_to_scale = tf.expand_dims(inp[:,...,0], axis = -1)
        
        out = self.stages[0](inp)
        for k in range(1, len(self.stages)):
            out = self.stages[k](out)
        out = tf.concat([self.spp(out), domain_info], axis = 1)
        out = self.dense_2(self.dense_1(self.dense_0(out)))
        return out
    
class Homogeneous_Poisson_NN_Fluidnet_Test(Model_With_Integral_Loss_ABC):
    '''
    Takes a tf.Tensor containing the RHS and grid spacing information, and gives the corresponding solution to the Poisson problem with homogeneous BCs.
    '''
    def __init__(self, pooling_block_number = 6, post_dx_einsum_conv_block_number = 5, initial_kernel_size = 19, final_kernel_size = 3, resize_methods = None, data_format = 'channels_first', use_batchnorm = False, use_deconv_upsample = False, kernel_regularizer = None, bias_regularizer = None, use_scaling = True, output_max_magnitude = 1.0, scaling_args = {'downsampling_ratio_per_stage': 3, 'stages': 3, 'filters': 4, 'activation': tf.nn.leaky_relu, 'spp_levels': [[2,2],3,5]}, **kwargs):
        '''
        Init arguments:

        data_format: same as keras
        pooling_block_number: integer. controls the number of simultaneous pooling 'threads'. each pooling thread will apply average pooling with pool_size 2^k (k = [1,...,pooling_block_number]), do convoltuions and then upsample to original resolution. choose such that 2^(pooling_block_number) is close as possible to the minimum grid size your inputs will have.
        use_deconv_upsample: boolean. if set to true, pooling blocks will use transposed convolution to upsample. last 2 pooling layers will use bilinear and nearest neighbor unless overridden by resize_methods
        resize_methods: list of tf.image.ResizeMethod members. used to choose resizing algorithms for pooling block upsampling.
        use_batchnorm: boolean. determines if batch norm layers should be used. if set to true, supply __call__ argument training = False when performing inference.
        post_dx_einsum_conv_block_number: integer. no of convolution blocks after the result of the domain info processing is merged.
        initial_kernel_size: integer. initial kernel size of convolutions. most convolutions will have this kernel size.
        final_kernel_size: integer. the kernel size of the final convolutions will progressively move towards this number. keep it small to prevent output artefacting near image edges.
        kernel_regularizer: same as the corresponding tf.keras.layers.Conv2D argument
        bias_regularizer: same as the corresponding tf.keras.layers.Conv2D argument

        **kwargs: used to set self.integral_loss parameters for the Model_With_Integral_Loss_ABC abstract base class
        '''
        super().__init__(**kwargs)
        self.training = True
        self.pooling_block_number = pooling_block_number
        self.data_format = data_format
        self.pooling_block_kernel_sizes = 9 * np.ones((self.pooling_block_number), dtype = np.int32)
        self.pooling_block_kernel_sizes[-2:] = 3
        self.pooling_block_kernel_sizes = list(self.pooling_block_kernel_sizes)
        self.use_batchnorm = use_batchnorm
        self.output_max_magnitude = output_max_magnitude

        final_dense_layer_units = 32
        
        if not resize_methods: #set resizing methods for pooling blocks if using bicubic upsampling
            try:
                self.resize_methods = [tf.compat.v1.image.ResizeMethod.BICUBIC for i in range(self.pooling_block_number-2)] + [tf.compat.v1.image.ResizeMethod.BILINEAR, tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR]
            except:
                self.resize_methods = [tf.image.ResizeMethod.BICUBIC for i in range(self.pooling_block_number-2)] + [tf.image.ResizeMethod.BILINEAR, tf.image.ResizeMethod.NEAREST_NEIGHBOR]
        else:
            self.resize_methods = resize_methods

        self.pooling_block_filters = 24
        
        self.conv_1 = tf.keras.layers.Conv2D(filters = 8, kernel_size = initial_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        #self.batchnorm_1 = tf.keras.layers.BatchNormalization(axis = 1)
        
        self.conv_2 = tf.keras.layers.Conv2D(filters = self.pooling_block_filters, kernel_size = initial_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        #self.batchnorm_2 = tf.keras.layers.BatchNormalization(axis = 1)
        
        self.conv_3 = tf.keras.layers.Conv2D(filters = self.pooling_block_filters, kernel_size = initial_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)

        if use_deconv_upsample:
            use_deconv_upsample = [True for k in range(self.pooling_block_number-2)] + [False for k in range(2)]
        else:
            use_deconv_upsample = [False for k in range(self.pooling_block_number)]
        self.pooling_blocks = [AveragePoolingBlock(2**(i+1), resize_method = self.resize_methods[i], data_format = data_format, kernel_size = int(self.pooling_block_kernel_sizes[i]), filters = self.pooling_block_filters, activation = tf.nn.leaky_relu, use_resnetblocks = True, use_deconv_upsample = use_deconv_upsample[i], kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer) for i in range(pooling_block_number)]
        
        self.merge = MergeWithAttention()
        
        self.dx_einsum_conv = tf.keras.layers.Conv2D(filters = final_dense_layer_units, kernel_size = initial_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        self.dx_einsum_resnet = ResnetBlock(filters = final_dense_layer_units, kernel_size = initial_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)

        self.post_dx_einsum_conv_blocks = []
        for k in range(post_dx_einsum_conv_block_number):
            ksize = initial_kernel_size + (k*(final_kernel_size - initial_kernel_size)//(post_dx_einsum_conv_block_number-1))
            filters = final_dense_layer_units-((k*final_dense_layer_units)//(post_dx_einsum_conv_block_number))
            block = []
            block.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = ksize, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            block.append(ResnetBlock(filters = filters, kernel_size = ksize, activation=tf.nn.leaky_relu, data_format=data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            self.post_dx_einsum_conv_blocks.append(block)

        self.residual_pooling_connection_blocks = []
        for k in range(3):
            block = []
            block.append(tf.keras.layers.Conv2D(filters = filters, kernel_size = final_kernel_size, activation = tf.nn.leaky_relu, data_format = data_format, padding = 'same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            block.append(ResnetBlock(filters = filters, kernel_size = final_kernel_size, activation = tf.nn.leaky_relu, data_format = data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
            self.residual_pooling_connection_blocks.append(block)

        if use_batchnorm:
            self.batchnorm_layers = []
            if self.data_format == 'channels_first':
                self.batchnorm_last = tf.keras.layers.BatchNormalization(axis = 1)
            else:
                self.batchnorm_last = tf.keras.layers.BatchNormalization(axis = -1)
            for k in range(post_dx_einsum_conv_block_number):
                if self.data_format == 'channels_first':
                    self.batchnorm_layers.append(tf.keras.layers.BatchNormalization(axis = 1))
                else:
                    self.batchnorm_layers.append(tf.keras.layers.BatchNormalization(axis = -1))

        self.conv_penultimate = tf.keras.layers.Conv2D(filters = filters, kernel_size = final_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        self.resnet_penultimate = ResnetBlock(filters = filters, kernel_size = final_kernel_size, activation=tf.nn.leaky_relu, data_format=data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)

        self.merge_final = MergeWithAttention()
        self.conv_last = tf.keras.layers.Conv2D(filters = 1, kernel_size = final_kernel_size, activation=tf.nn.tanh, data_format=data_format, padding='same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        self.resnet_last = ResnetBlock(filters = 1, kernel_size = final_kernel_size, activation=tf.nn.tanh, data_format=data_format, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        self.output_denoiser = AveragePoolingBlock(pool_size = 4, data_format=data_format, use_resnetblocks = True, use_deconv_upsample = False, kernel_size = final_kernel_size, filters = 1, activation=tf.nn.tanh, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer) #13.09 set activations to tanh
        
        self.dx_dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dx_dense_1 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dx_dense_2 = tf.keras.layers.Dense(final_dense_layer_units, activation = 'linear', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)


        self.use_scaling = use_scaling
        if use_scaling:
            #self.scaling = Scaling(data_format = self.data_format, kernel_size = final_kernel_size, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer, **scaling_args)

            self.dx_dense_scaler_0 = tf.keras.layers.Dense(30, activation = tf.nn.leaky_relu)
            self.dx_dense_scaler_1 = tf.keras.layers.Dense(30, activation = tf.nn.leaky_relu)
            self.dx_dense_scaler_2 = tf.keras.layers.Dense(1, activation = 'linear')
        
    def call(self, inp):
        '''
        input arguments:
        
        inp[0]: tf.Tensor of shape (batch_size, 1, nx, ny) or (batch_size, nx, ny, 1) based on self.data_format. this is the RHSes.
        inp[1]: tf.Tensor of shape (batch_size, 1). this must be the grid spacing information.
        '''
        self.dx = inp[1]
        if self.data_format == 'channels_first':#compute domain info
            domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[3]), tf.keras.backend.floatx())], axis = 0))
        else:
            domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[1]), tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx())], axis = 0))

        #inv_dom_size = tf.reduce_prod(1/domain_info[:,1:], axis = 1, keepdims = True)
        #domain_info = tf.concat([domain_info, inv_dom_size], axis = 1)
        
        out = self.conv_1(inp[0]) #initial convolutions
        out = self.conv_2(out)

        pooling_block_result = [pb(out) for pb in self.pooling_blocks]
        out = self.merge([self.conv_3(out)]+pooling_block_result)#pooling 'threads'

        out = self.dx_einsum_conv(out)
        out = self.dx_einsum_resnet(out)

        dx_res = self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(domain_info)))
        
        out = tf.einsum('ijkl, ij -> ijkl',out, dx_res) #incorporate domain info

        for block_num, block in enumerate(self.post_dx_einsum_conv_blocks): #perform additional convolutions
            for layer in block:
                out = layer(out)
            if self.use_batchnorm:
                out = self.batchnorm_layers[block_num](out, training = self.training)

        if self.use_batchnorm:
            out = self.batchnorm_last(out, training = self.training)
        out = self.resnet_penultimate(self.conv_penultimate(out))

        rpcb_res = []
        for blk_num, blk in enumerate(self.residual_pooling_connection_blocks):
            tmp = blk[0](pooling_block_result[blk_num])
            for lyr in blk[1:]:
                tmp = lyr(tmp)
            rpcb_res.append(tmp)

        out = self.merge_final([out]+rpcb_res)
        out = self.resnet_last(self.conv_last(out))
        #out += sum(rpcb_res)
        out = self.output_denoiser(out)
            
        if self.use_scaling:
            #out = self.scaling([out, inp[0], dx_res])
            try:
                scale = self.dx_dense_scaler_2(self.dx_dense_scaler_1(self.dx_dense_scaler_0(domain_info)))
                out = tf.einsum('i,i...->i...',tf.squeeze(scale),out)
            except:
                pass

        if self.training == False:
            out = tf.einsum('i...,i->i...', out, tf.reduce_prod(domain_info[:,1:], axis = 1))

        if self.output_max_magnitude is not None:
            out = smmib(out, 1.0)#13.09 added smmib

        return out

    def __call__(self, inp, training = True):#overload __call__ to allow freezing batch norm parameters
        self.training = training
        return super().__call__(inp)
