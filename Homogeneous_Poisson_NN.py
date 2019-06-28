import tensorflow as tf
import numpy as np
from Dirichlet_BC_NN import Model_With_Integral_Loss_ABC, ResampledConvolutionBlock, ResnetBlock
from MergeWithAttention import MergeWithAttention2
from Upsample import Upsample2
import itertools
import opt_einsum as oe

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

class AveragePoolingBlock(tf.keras.models.Model):
    def __init__(self, pool_size = 2, data_format = 'channels_first', resize_method = tf.image.ResizeMethod.BICUBIC, use_resnetblocks = False, **convblockargs):
        super().__init__()
        self.data_format = data_format
        self.pool = tf.keras.layers.AveragePooling2D(data_format = data_format, pool_size = pool_size, padding = 'same')
        self.upsample = Upsample2([-1,-1],resize_method=resize_method, data_format = data_format)

        if not use_resnetblocks:
            self.pooledconv = tf.keras.layers.Conv2D(data_format = data_format, padding='same', **convblockargs)
            self.upsampledconv = tf.keras.layers.Conv2D(data_format = data_format, padding='same', **convblockargs)
        else:
            self.pooledconv = ResnetBlock(data_format = data_format, **convblockargs)
            self.upsampledconv = ResnetBlock(data_format = data_format, **convblockargs)
    def call(self, inp):
        if self.data_format == 'channels_first':
            input_shape = [inp.shape[-2], inp.shape[-1]]
        else:
            input_shape = [inp.shape[-3],inp.shape[-2]]
        return self.upsampledconv(self.upsample([self.pooledconv(self.pool(inp)), input_shape]))
    
    
    
class Homogeneous_Poisson_NN_2(Model_With_Integral_Loss_ABC): #variant to include dx info
    def __init__(self, pooling_block_number = 6, resize_methods = None, data_format = 'channels_first', **kwargs):
        super().__init__(**kwargs)
        self.pooling_block_number = pooling_block_number
        self.data_format = data_format
        self.pooling_block_kernel_sizes = 3 * np.ones((self.pooling_block_number), dtype = np.int32)
        self.pooling_block_kernel_sizes[-2:] = 1
        self.pooling_block_kernel_sizes = list(self.pooling_block_kernel_sizes)
        
        if not resize_methods:
            try:
                self.resize_methods = [tf.compat.v1.image.ResizeMethod.BICUBIC for i in range(self.pooling_block_number-2)] + [tf.compat.v1.image.ResizeMethod.BILINEAR, tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR]
            except:
                self.resize_methods = [tf.image.ResizeMethod.BICUBIC for i in range(self.pooling_block_number-2)] + [tf.image.ResizeMethod.BILINEAR, tf.image.ResizeMethod.NEAREST_NEIGHBOR]
        else:
            self.resize_methods = resize_methods
        
        self.conv_1 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        
        self.pooling_blocks = [AveragePoolingBlock(2**(i+1), resize_method = self.resize_methods[i], data_format = data_format, kernel_size = int(self.pooling_block_kernel_sizes[i]), filters = 32, activation = tf.nn.leaky_relu, use_resnetblocks = False) for i in range(pooling_block_number)]
        
        self.merge = MergeWithAttention2()
        
        self.conv_4 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        self.conv_5 = tf.keras.layers.Conv2D(filters = 5, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        self.conv_6 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, activation='linear', data_format=data_format, padding='same')
        
        self.dx_dense_0 = tf.keras.layers.Dense(100, activation = tf.nn.relu)
        self.dx_dense_1 = tf.keras.layers.Dense(100, activation = tf.nn.relu)
        self.dx_dense_2 = tf.keras.layers.Dense(16, activation = 'linear')

        
    def call(self, inp):
        self.dx = inp[1]
        inp = inp[0]
        out = self.conv_2(self.conv_1(inp))
        out = self.merge([self.conv_3(out)] + [pb(out) for pb in self.pooling_blocks])
        return self.conv_6(self.conv_5(tf.einsum('ijkl, ij -> ijkl',self.conv_4(out), 1.0*(0.0+self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.dx)))))))

from SpatialPyramidPool import SpatialPyramidPool
class Homogeneous_Poisson_NN_3(Model_With_Integral_Loss_ABC): #variant to include dx info
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
        if self.data_format == 'channels_first':
            return tf.expand_dims(out, axis = 1)
        else:
            return tf.expand_dims(out, axis = 3)
