import tensorflow as tf
import numpy as np
import itertools
import opt_einsum as oe

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
    
class Homogeneous_Poisson_NN_Fluidnet(Model_With_Integral_Loss_ABC): #variant to include dx info
    def __init__(self, pooling_block_number = 6, post_dx_einsum_conv_block_number = 5, initial_kernel_size = 19, final_kernel_size = 3, resize_methods = None, data_format = 'channels_first', use_batchnorm = False, use_deconv_upsample = False, kernel_regularizer = None, bias_regularizer = None, **kwargs):
        super().__init__(**kwargs)
        self.training = True
        self.pooling_block_number = pooling_block_number
        self.data_format = data_format
        self.pooling_block_kernel_sizes = 9 * np.ones((self.pooling_block_number), dtype = np.int32)
        self.pooling_block_kernel_sizes[-2:] = 3
        self.pooling_block_kernel_sizes = list(self.pooling_block_kernel_sizes)
        self.use_batchnorm = use_batchnorm

        final_dense_layer_units = 32
        
        if not resize_methods:
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

        self.scaling = Scaling(downsampling_ratio_per_stage = 3, stages = 3, data_format = self.data_format, filters = 4, activation = tf.nn.leaky_relu, kernel_size = final_kernel_size, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        
    def call(self, inp):
        self.dx = inp[1]
        if self.data_format == 'channels_first':
            domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[3]), tf.keras.backend.floatx())], axis = 0))#, backend = 'tensorflow')
        else:
            domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[1]), tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx())], axis = 0))

        out = self.conv_1(inp[0])
        #out = self.batchnorm_1(out)
        
        out = self.conv_2(out)
        #out = self.batchnorm_2(out)
        
        out = self.merge([self.conv_3(out)] + [pb(out) for pb in self.pooling_blocks])

        #out = self.batchnorm_4(out)
        out = self.dx_einsum_conv(out)
        out = self.dx_einsum_resnet(out)
        
        out = tf.einsum('ijkl, ij -> ijkl',out, self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(domain_info))))

        for block_num, block in enumerate(self.post_dx_einsum_conv_blocks):
            for layer in block:
                out = layer(out)
            if self.use_batchnorm:
                out = self.batchnorm_layers[block_num](out, training = self.training)

        if self.use_batchnorm:
            out = self.batchnorm_last(out, training = self.training)
        out = self.conv_last(out)
        out = self.resnet_last(out)
        
        return self.scaling([out, inp[0]])

    def __call__(self, inp, training = True):
        self.training = training
        return super().__call__(inp)
    

class Homogeneous_Poisson_NN_Fourier(Model_With_Integral_Loss_ABC): #variant to include dx info
    def __init__(self, resnet_block_number = 6, filters = [8,16,32,64,16,8], kernel_sizes = [5,5,5,5,5,5], nmodes = (32,32), pyramid_pooling_params = {'levels': [[3,3],6,9,12], 'pooling_type': 'AVG'}, data_format = 'channels_first', **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format
        self.resnet_blocks = []
        self.nmodes = nmodes
        
        for i in range(resnet_block_number):
            self.resnet_blocks.append(tf.keras.layers.Conv2D(filters = filters[i], kernel_size = kernel_sizes[i], data_format = self.data_format, activation = tf.nn.leaky_relu, padding = 'same'))
            self.resnet_blocks.append(ResnetBlock(filters = filters[i], kernel_size = kernel_sizes[i], data_format = self.data_format, activation = tf.nn.leaky_relu))
        self.spp = SpatialPyramidPool(data_format = self.data_format, **pyramid_pooling_params)
        
        self.dx_dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dx_dense_1 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dx_dense_2 = tf.keras.layers.Dense(16, activation = tf.nn.leaky_relu)
        
        self.fourier_dense_0 = tf.keras.layers.Dense(250, activation = tf.nn.leaky_relu)
        self.fourier_dense_1 = tf.keras.layers.Dense(500, activation = tf.nn.leaky_relu)
        self.fourier_dense_2 = tf.keras.layers.Dense(tf.reduce_prod(nmodes), activation = lambda x: 4*tf.nn.tanh(x)/np.pi**2)

        #self.scaling = Scaling()
        
        m,n = tf.meshgrid(np.arange(self.nmodes[0])+1,np.arange(self.nmodes[1])+1)
        self.m = tf.expand_dims((tf.cast(m, tf.keras.backend.floatx())**2) * (tf.cast(np.pi, tf.keras.backend.floatx())**2), axis = 0)
        self.n = tf.expand_dims((tf.cast(n, tf.keras.backend.floatx())**2) * (tf.cast(np.pi, tf.keras.backend.floatx())**2), axis = 0)
                
    def call(self, inputs):
        self.dx = inputs[1]
        inp = inputs[0]

        #compute domain shape and extract domain info
        try: #stupid hack 1 to get past keras '?' tensor dimensions via try except block
            if self.data_format == 'channels_first':
                self.domain_info = oe.contract('ij,j->ij',tf.tile(inputs[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[2]), tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[3]), tf.keras.backend.floatx())], axis = 0), backend = 'tensorflow')
                dx_res = self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info)))
                X,Y = tf.meshgrid(tf.linspace(0.0,1.0,inputs[0].shape[2]), tf.linspace(0.0,1.0,inputs[0].shape[3]), indexing = 'ij')
            else:
                self.domain_info = oe.contract('ij,j->ij',tf.tile(inputs[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[1]), tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[2]), tf.keras.backend.floatx())], axis = 0), backend = 'tensorflow')
                dx_res = self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info)))
                X,Y = tf.meshgrid(tf.linspace(0.0,1.0,inputs[0].shape[1]), tf.linspace(0.0,1.0,inputs[0].shape[2]), indexing = 'ij')            
            X = tf.cast(X, tf.keras.backend.floatx())
            Y = tf.cast(Y, tf.keras.backend.floatx())
            sine_values = tf.Variable(tf.zeros(list(self.nmodes) + list(X.shape), dtype = tf.keras.backend.floatx()), dtype = tf.keras.backend.floatx())
            for i in range(self.nmodes[0]):
                for j in range(self.nmodes[1]):
                    sine_values[i,j,...].assign(tf.sin(((i+1)*np.pi)*X) * tf.sin(((j+1)*np.pi)*Y))
        except:
            print('Initializing model.')
            return inputs[0]+1

        #Compute Fourier series amplitudes
        amplitudes = self.resnet_blocks[0](inputs[0])
        for layer in self.resnet_blocks:
            amplitudes = layer(amplitudes)
        amplitudes = tf.concat([self.spp(amplitudes),dx_res], axis = 1)
        amplitudes = oe.contract('ijk,i->ijk',tf.reshape(self.fourier_dense_2(self.fourier_dense_1(self.fourier_dense_0(amplitudes))), [-1] + list(self.nmodes)), tf.reduce_prod(self.domain_info[:,1:], axis = 1), backend = 'tensorflow')
        
        LxoverLy = self.domain_info[:,1]/self.domain_info[:,2]
        m = oe.contract('ijk,i->ijk', tf.tile(self.m, [inputs[0].shape[0],1,1]), 1/LxoverLy, backend = 'tensorflow')
        n = oe.contract('ijk,i->ijk', tf.tile(self.n, [inputs[0].shape[0],1,1]), LxoverLy, backend = 'tensorflow')
        amplitudes = amplitudes*(-4.0)/(m+n)
        #Use the fourier coefficients to recover the solutions
        out = oe.contract('bxy,xyij->bij', amplitudes, sine_values, backend = 'tensorflow')
        #out = self.scaling(out)
        if self.data_format == 'channels_first':
            return tf.expand_dims(out, axis = 1)
        else:
            return tf.expand_dims(out, axis = 3)
