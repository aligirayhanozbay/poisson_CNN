import tensorflow as tf
import numpy as np
import itertools
import opt_einsum as oe
import string
import copy

from .custom_blocks import ResampledConvolutionBlock, ResnetBlock, AveragePoolingBlock, Scaling
from .Model_With_Integral_Loss import Model_With_Integral_Loss_ABC
from ..layers import MergeWithAttention
from ..layers import Upsample
from ..layers import SpatialPyramidPool
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
    
class Homogeneous_Poisson_NN_Fluidnet_IA(Model_With_Integral_Loss_ABC):
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
        ia = []
        self.dx = inp[1]
        if self.data_format == 'channels_first':#compute domain info
            domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[3]), tf.keras.backend.floatx())], axis = 0))
        else:
            domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[1]), tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx())], axis = 0))

        out = self.conv_1(inp[0]) #initial convolutions
        ia.append(copy.deepcopy(out))
        out = self.conv_2(out)
        ia.append(copy.deepcopy(out))
        for pb in self.pooling_blocks:
            ia.append(pb(out))
        ia.append(self.conv_3(out))
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

        ia.append(out)
        return ia

    def __call__(self, inp, training = True):#overload __call__ to allow freezing batch norm parameters
        self.training = training
        return super().__call__(inp)
