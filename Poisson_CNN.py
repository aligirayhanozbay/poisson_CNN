import tensorflow as tf
import numpy as np
import opt_einsum as oe
import itertools
from Homogeneous_Poisson_NN import Homogeneous_Poisson_NN_2
from Dirichlet_BC_NN import Dirichlet_BC_NN_2D, Model_With_Integral_Loss_ABC
#tf.keras.backend.set_floatx('float64')

def channels_first_rot90(image,k=1):
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
    
def channels_first_flip_up_down(image):
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
    
    image = tf.image.flip_left_right(image) #tensorflow seems to have the opposite alignment...
    
    if len(image.shape) == 4:
        return tf.transpose(image, (0,3,1,2))
    elif image_was_rank3:
        return tf.transpose(image, (0,2,1))
    else:
        return image[...,0]
    
def channels_first_flip_left_right(image):
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
    
    image = tf.image.flip_up_down(image) #tensorflow seems to have the opposite alignment...
    
    if len(image.shape) == 4:
        return tf.transpose(image, (0,3,1,2))
    elif image_was_rank3:
        return tf.transpose(image, (0,2,1))
    else:
        return image[...,0]

class Poisson_CNN(Model_With_Integral_Loss_ABC):
    def __init__(self, bc_nn_parameters = {}, homogeneous_poisson_nn_parameters = {}, bc_nn_weights = None,  homogeneous_poisson_nn_weights = None, bc_nn_trainable = True, homogeneous_poisson_nn_trainable = True, data_format = 'channels_first', **kwargs):
        super().__init__(**kwargs)
        
        self.data_format = data_format
        if self.data_format == 'channels_first':
            self.rotation_method = channels_first_rot90
            self.fliplr_method = channels_first_flip_left_right
            self.flipud_method = channels_first_flip_up_down
            self.stacking_dim = 1
        else:
            self.rotation_method = tf.image.rot90
            self.fliplr_method = tf.image.flip_up_down
            self.flipud_method = tf.image.flip_left_right
            self.stacking_dim = 3
        if 'data_format' not in bc_nn_parameters.keys():
            bc_nn_parameters['data_format'] = self.data_format
        if 'data_format' not in homogeneous_poisson_nn_parameters.keys():
            homogeneous_poisson_nn_parameters['data_format'] = self.data_format
        
        self.bc_nn = Dirichlet_BC_NN_2D(**bc_nn_parameters)
        self.bc_nn((tf.random.uniform((10,1,74), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1), dtype = tf.keras.backend.floatx())))
        try:
            self.bc_nn.load_weights(bc_nn_weights)
        except:
            pass
        for layer in self.bc_nn.layers:
            layer.trainable = bc_nn_trainable
        self.post_bc_nn_conv_0 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, activation = tf.nn.leaky_relu, data_format = self.data_format, padding = 'same')
        
        self.hpnn = Homogeneous_Poisson_NN_2(**homogeneous_poisson_nn_parameters)
        self.hpnn((tf.random.uniform((10,1,74,83), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1), dtype = tf.keras.backend.floatx())))
        try:
            self.hpnn.load_weights(homogeneous_poisson_nn_weights)
        except:
            pass
        for layer in self.hpnn.layers:
            layer.trainable = homogeneous_poisson_nn_trainable
        self.post_HPNN_conv_0 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, activation = tf.nn.leaky_relu, data_format = self.data_format, padding = 'same')
        
        self.post_merge_convolution_0 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, activation = tf.nn.leaky_relu, data_format = self.data_format, padding = 'same')
        self.post_merge_convolution_1 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 5, activation = tf.nn.leaky_relu, data_format = self.data_format, padding = 'same')
        self.post_merge_convolution_2 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 5, activation = 'linear', data_format = self.data_format, padding = 'same')
        
    def call(self, inp):# rhs, boundaries, dx):
        #inp format - inp[0]: rhs array (tf.Tensor) | inp[1] boundaries (dict) | inp[2] dx (tf.Tensor)
        #implement bc nn parameter adjustment to permit any output shape
        rhs = inp[0]
        left_boundary = inp[1]
        top_boundary = inp[2]
        right_boundary = inp[3]
        bottom_boundary = inp[4]
        self.dx = inp[5]
        if len(self.dx.shape) == 1:
            self.dx = tf.expand_dims(self.dx,axis = 1)
        if self.data_format == 'channels_first':
            self.bc_nn.x_output_resolution = rhs.shape[2]
        else:
            self.bc_nn.x_output_resolution = rhs.shape[1]
        laplace_soln = self.bc_nn([left_boundary, self.dx]) + self.flipud_method(self.rotation_method(self.bc_nn([right_boundary, self.dx]), k = 2))
        if self.data_format == 'channels_first':
            self.bc_nn.x_output_resolution = rhs.shape[3]
        else:
            self.bc_nn.x_output_resolution = rhs.shape[2]
        laplace_soln = self.post_bc_nn_conv_0(laplace_soln + self.fliplr_method(self.rotation_method(self.bc_nn([top_boundary, self.dx]), k = 1)) + self.rotation_method(self.bc_nn([bottom_boundary, self.dx]), k = 3))
        #         laplace_soln = self.post_bc_nn_conv_0(self.bc_nn([left_boundary, self.dx]) + self.flipud_method(self.rotation_method(self.bc_nn([right_boundary, self.dx]), k = 2)) + self.fliplr_method(self.rotation_method(self.bc_nn([top_boundary, self.dx]), k = 1)) + self.rotation_method(self.bc_nn([bottom_boundary, self.dx]), k = 3))
        homogeneous_poisson_soln = self.post_HPNN_conv_0(self.hpnn([rhs, self.dx]))
        #         print(laplace_soln.shape)
        #         print(homogeneous_poisson_soln.shape)
        #         return homogeneous_poisson_soln + laplace_soln
        return self.post_merge_convolution_2(self.post_merge_convolution_1(self.post_merge_convolution_0(homogeneous_poisson_soln + laplace_soln)))