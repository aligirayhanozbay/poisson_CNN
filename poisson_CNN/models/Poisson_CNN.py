import tensorflow as tf
import numpy as np
import opt_einsum as oe
import itertools


from .Homogeneous_Poisson_NN import *
from .Dirichlet_BC_NN import *
from .custom_blocks import *
from ..dataset.generators.numerical import set_max_magnitude_in_batch as smmib

#helper methods to do rotation/reflection operations with data_format = 'channels_first'
def channels_first_rot90(image,k=1):
    if len(image.shape) == 4:
        image = tf.transpose(image, (0,2,3,1))
    elif len(image.shape) == 3:
        image = tf.transpose(image, (1,2,0))
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
        return tf.transpose(image, (2,0,1))
    else:
        return image[...,0]
    
def channels_first_flip_up_down(image):
    if len(image.shape) == 4:
        image = tf.transpose(image, (0,2,3,1))
    elif len(image.shape) == 3:
        image = tf.transpose(image, (1,2,0))
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
        return tf.transpose(image, (2,0,1))
    else:
        return image[...,0]
    
def channels_first_flip_left_right(image):
    if len(image.shape) == 4:
        image = tf.transpose(image, (0,2,3,1))
    elif len(image.shape) == 3:
        image = tf.transpose(image, (1,2,0))
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
        return tf.transpose(image, (2,0,1))
    else:
        return image[...,0]

class Poisson_CNN(Model_With_Integral_Loss_ABC):
    '''
    A CNN model that predicts the solution of a Poisson problem with Dirichlet BCs
    '''
    def __init__(self, bc_nn_parameters = {}, homogeneous_poisson_nn_parameters = {}, bc_nn_weights = None,  homogeneous_poisson_nn_weights = None, bc_nn_trainable = True, homogeneous_poisson_nn_trainable = True, data_format = 'channels_first', **kwargs):
        '''
        Init arguments:

        bc_nn_parameters: Dict containing arguments to be supplied to Dirichlet_BC_NN as Dirichlet_BC_NN(**bc_nn_parameters)
        homogeneous_poisson_nn_parameters: Dict containing arguments to be supplied to Homogeneous_Poisson_NN
        bc_nn_weights: String or None. If a string, it must be the (relative) filepath to a HDF5 file contanining the weights for Dirichlet_BC_NN. No weights will be loaded if left as None.
        homogeneous_poisson_nn_weights: String or None. If a string, it must be the (relative) filepath to a HDF5 file contanining the weights for Homogeneous_Poisson_NN. No weights will be loaded if left as None.
        bc_nn_trainable: If set to True, BC NN weights will not be frozen during training.
        homogeneous_poisson_nn_trainable: If set to True, Homogeneous_Poisson_NN weighrs will not be frozen during training.
        data_format: Same as tf.keras.layers.Conv2D
        
        **kwargs: used to set self.integral_loss parameters for the Model_With_Integral_Loss_ABC abstract base class
        '''
        super().__init__(**kwargs)

        #choose image reflection/rotation functions
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

        #instantiate Dirichlet_BC_NN, initialize weights, load weight file
        self.bc_nn = Dirichlet_BC_NN(**bc_nn_parameters)
        self.bc_nn((tf.random.uniform((10,1,74), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1), dtype = tf.keras.backend.floatx()))) #initialize weights
        try:
            self.bc_nn.load_weights(bc_nn_weights)
        except:
            pass
        for layer in self.bc_nn.layers:
            layer.trainable = bc_nn_trainable

        #instantiate Homogeneous_Poisson_NN, initialize weights, load weight file
        self.hpnn = Homogeneous_Poisson_NN_Fluidnet(**homogeneous_poisson_nn_parameters)
        self.hpnn((tf.random.uniform((10,1,74,83), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1), dtype = tf.keras.backend.floatx())))
        try:
            self.hpnn.load_weights(homogeneous_poisson_nn_weights)
        except:
            pass

        for layer in self.hpnn.layers:
            layer.trainable = homogeneous_poisson_nn_trainable

        
    def call(self, inp):# rhs, boundaries, dx):
        '''
        Call arguments:

        inp[0]: tf.Tensor of shape (batch_size, 1, nx, ny) or (batch_size, nx, ny, 1) based on self.data_format. this is the RHSes.
        inp[1]: tf.Tensor of shape (batch_size, 1, ny) or (batch_size, ny, 1) based on self.data_format. this is the left Dirichlet BC.
        inp[2]: tf.Tensor of shape (batch_size, 1, nx) or (batch_size, nx, 1) based on self.data_format. this is the top Dirichlet BC.
        inp[3]: tf.Tensor of shape (batch_size, 1, ny) or (batch_size, ny, 1) based on self.data_format. this is the right Dirichlet BC.
        inp[4]: tf.Tensor of shape (batch_size, 1, nx) or (batch_size, nx, 1) based on self.data_format. this is the bottom Dirichlet BC.
        inp[5]: tf.Tensor of shape (batch_size, 1). this must be the grid spacing information.
        '''

        #scale all inputs to have 1.0 peak magnitudes
        rhs, rhs_scaling_factors = smmib(inp[0], 1.0, return_scaling_factors = True)
        left_boundary, left_boundary_scaling_factors = smmib(inp[1], 1.0, return_scaling_factors = True)
        top_boundary, top_boundary_scaling_factors = smmib(inp[2], 1.0, return_scaling_factors = True)
        right_boundary, right_boundary_scaling_factors = smmib(inp[3], 1.0, return_scaling_factors = True)
        bottom_boundary, bottom_boundary_scaling_factors = smmib(inp[4], 1.0, return_scaling_factors = True)
        scaling_factors = tf.stack([left_boundary_scaling_factors, right_boundary_scaling_factors, top_boundary_scaling_factors, bottom_boundary_scaling_factors, rhs_scaling_factors], axis = 1)
        
        self.dx = inp[5]
        if len(self.dx.shape) == 1:
            self.dx = tf.expand_dims(self.dx,axis = 1)

        if self.data_format == 'channels_first':
            self.bc_nn.x_output_resolution = rhs.shape[2]#predict for left and right BCs by supplying them to Dirichlet_BC_NN as a single batch
            tmp = self.bc_nn([tf.concat([left_boundary, right_boundary], axis = 0), tf.tile(self.dx, [2,1])])
            left_boundary = tmp[0:rhs.shape[0],0,...]
            right_boundary = self.flipud_method(self.rotation_method(tmp[rhs.shape[0]:,0,...], k = 2)) #rotate and reflect right boundary method

            self.bc_nn.x_output_resolution = rhs.shape[3]#predict for top and bottom BCs by supplying them to Dirichlet_BC_NN as a single batch
            tmp = self.bc_nn([tf.concat([bottom_boundary, top_boundary], axis = 0), tf.tile(self.dx, [2,1])])
            bottom_boundary = self.fliplr_method(self.rotation_method(tmp[0:rhs.shape[0],0,...], k = 1)) #rotate and reflect top boundary prediction
            top_boundary = self.rotation_method(tmp[rhs.shape[0]:,0,...], k = 3) #rotate bottom boundary prediction

            rhs = tf.squeeze(self.hpnn([rhs, self.dx]), axis = self.stacking_dim) #predict for the homogeneous Poisson problem

            solutions = tf.stack([left_boundary, right_boundary, top_boundary, bottom_boundary, rhs], axis = 1) 
            solutions = tf.einsum('ij...,ij->ij...', solutions, 1/scaling_factors) #revert each prediction to original scale
            
        else: #same procedures but for channels_last
            self.bc_nn.x_output_resolution = rhs.shape[2]
            tmp = self.bc_nn([tf.concat([left_boundary, right_boundary], axis = 0), tf.tile(self.dx, [2,1])])
            left_boundary = tmp[0:rhs.shape[0],...,0]
            right_boundary = self.flipud_method(self.rotation_method(tmp[rhs.shape[0]:,...,0], k = 2))

            self.bc_nn.x_output_resolution = rhs.shape[3]
            tmp = self.bc_nn([tf.concat([top_boundary, bottom_boundary], axis = 0), tf.tile(self.dx, [2,1])])
            top_boundary = self.fliplr_method(self.rotation_method(tmp[0:rhs.shape[0],...,0], k = 1))
            bottom_boundary = self.rotation_method(tmp[rhs.shape[0]:,...,0], k = 3)

            rhs = tf.squeeze(self.hpnn([rhs, self.dx]))

            solutions = tf.stack([left_boundary, right_boundary, top_boundary, bottom_boundary, rhs], axis = -1)
            solutions = tf.einsum('i...j,ij->i...j', solutions, 1/scaling_factors)
        
        return tf.reduce_sum(solutions, axis = self.stacking_dim, keepdims = True)

class Poisson_CNN_Test(Model_With_Integral_Loss_ABC):
    '''
    A CNN model that predicts the solution of a Poisson problem with Dirichlet BCs
    '''
    def __init__(self, bc_nn_parameters = {}, homogeneous_poisson_nn_parameters = {}, bc_nn_weights = None,  homogeneous_poisson_nn_weights = None, bc_nn_trainable = True, homogeneous_poisson_nn_trainable = True, data_format = 'channels_first', **kwargs):
        '''
        Init arguments:

        bc_nn_parameters: Dict containing arguments to be supplied to Dirichlet_BC_NN as Dirichlet_BC_NN(**bc_nn_parameters)
        homogeneous_poisson_nn_parameters: Dict containing arguments to be supplied to Homogeneous_Poisson_NN
        bc_nn_weights: String or None. If a string, it must be the (relative) filepath to a HDF5 file contanining the weights for Dirichlet_BC_NN. No weights will be loaded if left as None.
        homogeneous_poisson_nn_weights: String or None. If a string, it must be the (relative) filepath to a HDF5 file contanining the weights for Homogeneous_Poisson_NN. No weights will be loaded if left as None.
        bc_nn_trainable: If set to True, BC NN weights will not be frozen during training.
        homogeneous_poisson_nn_trainable: If set to True, Homogeneous_Poisson_NN weighrs will not be frozen during training.
        data_format: Same as tf.keras.layers.Conv2D
        
        **kwargs: used to set self.integral_loss parameters for the Model_With_Integral_Loss_ABC abstract base class
        '''
        super().__init__(**kwargs)

        #choose image reflection/rotation functions
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

        #instantiate Dirichlet_BC_NN, initialize weights, load weight file
        self.bc_nn = Dirichlet_BC_NN(**bc_nn_parameters)
        self.bc_nn((tf.random.uniform((10,1,74), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1), dtype = tf.keras.backend.floatx()))) #initialize weights
        try:
            self.bc_nn.load_weights(bc_nn_weights)
        except:
            pass
        for layer in self.bc_nn.layers:
            layer.trainable = bc_nn_trainable

        #instantiate Homogeneous_Poisson_NN, initialize weights, load weight file
        self.hpnn = Homogeneous_Poisson_NN_Fluidnet(**homogeneous_poisson_nn_parameters)
        self.hpnn((tf.random.uniform((10,1,74,83), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1), dtype = tf.keras.backend.floatx())))
        try:
            self.hpnn.load_weights(homogeneous_poisson_nn_weights)
        except:
            pass
        for layer in self.hpnn.layers:
            layer.trainable = homogeneous_poisson_nn_trainable

        self.output_denoiser = self.output_denoiser = AveragePoolingBlock(pool_size = 4, data_format=data_format, use_resnetblocks = True, use_deconv_upsample = False, kernel_size = final_kernel_size, filters = 1, activation=tf.nn.tanh, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        
        
    def call(self, inp):# rhs, boundaries, dx):
        '''
        Call arguments:

        inp[0]: tf.Tensor of shape (batch_size, 1, nx, ny) or (batch_size, nx, ny, 1) based on self.data_format. this is the RHSes.
        inp[1]: tf.Tensor of shape (batch_size, 1, ny) or (batch_size, ny, 1) based on self.data_format. this is the left Dirichlet BC.
        inp[2]: tf.Tensor of shape (batch_size, 1, nx) or (batch_size, nx, 1) based on self.data_format. this is the top Dirichlet BC.
        inp[3]: tf.Tensor of shape (batch_size, 1, ny) or (batch_size, ny, 1) based on self.data_format. this is the right Dirichlet BC.
        inp[4]: tf.Tensor of shape (batch_size, 1, nx) or (batch_size, nx, 1) based on self.data_format. this is the bottom Dirichlet BC.
        inp[5]: tf.Tensor of shape (batch_size, 1). this must be the grid spacing information.
        '''

        #scale all inputs to have 1.0 peak magnitudes
        rhs, rhs_scaling_factors = smmib(inp[0], 1.0, return_scaling_factors = True)
        left_boundary, left_boundary_scaling_factors = smmib(inp[1], 1.0, return_scaling_factors = True)
        top_boundary, top_boundary_scaling_factors = smmib(inp[2], 1.0, return_scaling_factors = True)
        right_boundary, right_boundary_scaling_factors = smmib(inp[3], 1.0, return_scaling_factors = True)
        bottom_boundary, bottom_boundary_scaling_factors = smmib(inp[4], 1.0, return_scaling_factors = True)
        scaling_factors = tf.stack([left_boundary_scaling_factors, right_boundary_scaling_factors, top_boundary_scaling_factors, bottom_boundary_scaling_factors, rhs_scaling_factors], axis = 1)
        
        self.dx = inp[5]
        if len(self.dx.shape) == 1:
            self.dx = tf.expand_dims(self.dx,axis = 1)

        if self.data_format == 'channels_first':
            self.bc_nn.x_output_resolution = rhs.shape[2]#predict for left and right BCs by supplying them to Dirichlet_BC_NN as a single batch
            tmp = self.bc_nn([tf.concat([left_boundary, right_boundary], axis = 0), tf.tile(self.dx, [2,1])])
            left_boundary = tmp[0:rhs.shape[0],0,...]
            right_boundary = self.flipud_method(self.rotation_method(tmp[rhs.shape[0]:,0,...], k = 2)) #rotate and reflect right boundary method

            self.bc_nn.x_output_resolution = rhs.shape[3]#predict for top and bottom BCs by supplying them to Dirichlet_BC_NN as a single batch
            tmp = self.bc_nn([tf.concat([bottom_boundary, top_boundary], axis = 0), tf.tile(self.dx, [2,1])])
            bottom_boundary = self.fliplr_method(self.rotation_method(tmp[0:rhs.shape[0],0,...], k = 1)) #rotate and reflect top boundary prediction
            top_boundary = self.rotation_method(tmp[rhs.shape[0]:,0,...], k = 3) #rotate bottom boundary prediction

            rhs = tf.squeeze(self.hpnn([rhs, self.dx]), axis = self.stacking_dim) #predict for the homogeneous Poisson problem

            solutions = tf.stack([left_boundary, right_boundary, top_boundary, bottom_boundary, rhs], axis = 1) 
            solutions = tf.einsum('ij...,ij->ij...', solutions, 1/scaling_factors) #revert each prediction to original scale
            
        else: #same procedures but for channels_last
            self.bc_nn.x_output_resolution = rhs.shape[2]
            tmp = self.bc_nn([tf.concat([left_boundary, right_boundary], axis = 0), tf.tile(self.dx, [2,1])])
            left_boundary = tmp[0:rhs.shape[0],...,0]
            right_boundary = self.flipud_method(self.rotation_method(tmp[rhs.shape[0]:,...,0], k = 2))

            self.bc_nn.x_output_resolution = rhs.shape[3]
            tmp = self.bc_nn([tf.concat([top_boundary, bottom_boundary], axis = 0), tf.tile(self.dx, [2,1])])
            top_boundary = self.fliplr_method(self.rotation_method(tmp[0:rhs.shape[0],...,0], k = 1))
            bottom_boundary = self.rotation_method(tmp[rhs.shape[0]:,...,0], k = 3)

            rhs = tf.squeeze(self.hpnn([rhs, self.dx]))

            solutions = tf.stack([left_boundary, right_boundary, top_boundary, bottom_boundary, rhs], axis = -1)
            solutions = tf.einsum('i...j,ij->i...j', solutions, 1/scaling_factors)
        
        return self.output_denoiser(tf.reduce_sum(solutions, axis = self.stacking_dim, keepdims = True))
