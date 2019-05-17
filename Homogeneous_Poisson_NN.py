import tensorflow as tf
import numpy as np
from Dirichlet_BC_NN import Model_With_Integral_Loss_ABC
from MergeWithAttention import MergeWithAttention2
from Upsample import Upsample2
from Lp_integral_norm import Lp_integral_norm
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
    def __init__(self, pool_size = 2, data_format = 'channels_first', filters = 8, activation = tf.nn.leaky_relu, resize_method = tf.image.ResizeMethod.BICUBIC, kernel_size = 3):
        super().__init__()
        self.data_format = data_format
        self.pool = tf.keras.layers.AveragePooling2D(data_format = data_format, pool_size = pool_size)
        self.pooledconv = tf.keras.layers.Conv2D(filters = 8, kernel_size = int(kernel_size), activation=activation, data_format = data_format, padding='same')
        self.upsample = Upsample2([-1,-1],resize_method=resize_method, data_format = data_format)
        self.upsampledconv = tf.keras.layers.Conv2D(filters = 8, kernel_size = int(kernel_size), activation=activation, data_format = data_format, padding='same')
        
    def call(self, inp):
        if self.data_format == 'channels_first':
            input_shape = [inp.shape[-2], inp.shape[-1]]
        else:
            input_shape = [inp.shape[-3],inp.shape[-2]]
        return self.upsampledconv(self.upsample([self.pooledconv(self.pool(inp)), input_shape]))


class Homogeneous_Poisson_NN(tf.keras.models.Model):
    def __init__(self, pooling_block_number = 6, resize_methods = None, data_format = 'channels_first'):
        super().__init__()
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
        
        self.conv_1 = tf.keras.layers.Conv2D(filters = 5, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        
        self.pooling_blocks = [AveragePoolingBlock(pool_size = (2**(i+1)), resize_method = self.resize_methods[i], data_format = data_format, kernel_size = self.pooling_block_kernel_sizes[i]) for i in range(pooling_block_number)]
        
        self.merge = MergeWithAttention2()
        
        self.conv_4 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        self.conv_5 = tf.keras.layers.Conv2D(filters = 5, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        self.conv_6 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, activation='linear', data_format=data_format, padding='same')
        
    def call(self, inp):
        
        out = self.conv_2(self.conv_1(inp))
        out = self.merge([self.conv_3(out)] + [pb(out) for pb in self.pooling_blocks])
        return self.conv_6(self.conv_5(self.conv_4(out)))
    
    
    
    
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
        
        self.conv_1 = tf.keras.layers.Conv2D(filters = 5, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        self.conv_2 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        self.conv_3 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, activation=tf.nn.leaky_relu, data_format=data_format, padding='same')
        
        self.pooling_blocks = [AveragePoolingBlock(pool_size = (2**(i+1)), resize_method = self.resize_methods[i], data_format = data_format, kernel_size = self.pooling_block_kernel_sizes[i]) for i in range(pooling_block_number)]
        
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
