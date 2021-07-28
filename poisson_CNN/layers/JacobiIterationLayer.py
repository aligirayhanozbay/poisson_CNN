import tensorflow as tf
import numpy as np
from ..dataset.utils import build_fd_coefficients
from ..utils import choose_conv_layer, convert_keras_dataformat_to_tf
from ..layers.metalearning_conv import convolution_and_bias_add_closure

class JacobiIterationLayer(tf.keras.layers.Layer):
    def __init__(self, stencil_sizes, orders, ndims = None, data_format = 'channels_first', n_iterations = 5):
        super().__init__()
        coefficients = build_fd_coefficients(stencil_sizes, orders, ndims = ndims)
        diagonal_value_coefficients_slice = (Ellipsis,) + tuple([stencil_size//2 for stencil_size in stencil_sizes])
        self.diagonal_coefficients = tf.constant(coefficients[diagonal_value_coefficients_slice], dtype = tf.keras.backend.floatx())
        coefficients[diagonal_value_coefficients_slice] = np.zeros(*coefficients[diagonal_value_coefficients_slice].shape)
        self.LU_coefficients = tf.constant(coefficients, dtype = tf.keras.backend.floatx())
        
        self.ndims = coefficients.shape[0]
        self.data_format = data_format
        self._tf_data_format = convert_keras_dataformat_to_tf(self.data_format, self.ndims)
        self.orders = tf.constant(orders, dtype = tf.keras.backend.floatx())
        self.stencil_sizes = tf.constant(stencil_sizes)
        self.n_iterations = n_iterations
        
        cm = eval('tf.nn.conv' + str(self.ndims) + 'd')
        conv_method = lambda *args,**kwargs: cm(*args, strides = [1 for _ in range(self.ndims + 2)], dilations = [1 for _ in range(self.ndims + 2)], **kwargs)
        self.conv_method = convolution_and_bias_add_closure(self._tf_data_format, conv_method, use_bias = False)
        
        self.rhs_slice = (Ellipsis,) + tuple([slice(stencil_size//2,-(stencil_size//2)) for stencil_size in stencil_sizes]) + tuple([slice(0,None)] if self.data_format == 'channels_last' else [])
        if self.data_format == 'channels_first':
            self.update_mask_paddings = [[0,0],[0,0]] + [[stencil_size//2,stencil_size//2] for stencil_size in stencil_sizes]
        else:
            self.update_mask_paddings = [[0,0]] + [[stencil_size//2,stencil_size//2] for stencil_size in stencil_sizes] + [[0,0]]

    @tf.function
    def build_lu_conv_kernel(self, dxp):
        lu_conv_kernel = tf.einsum('d...,bd->b...', self.LU_coefficients, dxp)
        lu_conv_kernel = tf.expand_dims(tf.expand_dims(lu_conv_kernel,-1),-1)
        return lu_conv_kernel

    @tf.function
    def build_diagonal_coefficient(self, dxp):
        return 1/tf.einsum('bd,d->b',dxp,self.diagonal_coefficients)

    @tf.function
    def convolutional_jacobi_iteration(self, current_guess, rhses, lu_conv_kernel, d_inverse):
        #import pdb
        #pdb.set_trace()
        cr = self.conv_method(current_guess, lu_conv_kernel)
        new_guess = tf.einsum('b,b...->b...',d_inverse,rhses[self.rhs_slice] - cr)
        new_guess = tf.pad(new_guess, self.update_mask_paddings, mode='CONSTANT', constant_values = tf.constant(0.0,dtype = tf.keras.backend.floatx()))
        update_mask = tf.zeros(tf.shape(rhses[self.rhs_slice]), dtype = tf.keras.backend.floatx())
        update_mask = tf.pad(update_mask, self.update_mask_paddings, mode='CONSTANT', constant_values = tf.constant(1.0,dtype = tf.keras.backend.floatx()))
        new_guess = new_guess + current_guess * update_mask
        return new_guess

    @tf.function
    def call(self, inp):
        current_guess, rhses, dx = inp
        dxp = tf.map_fn(lambda x: tf.pow(x, self.orders), 1/dx) #(1/dx[k])**orders[k]
        lu_conv_kernel = self.build_lu_conv_kernel(dxp)
        d_inverse = self.build_diagonal_coefficient(dxp)

        new_guess = self.convolutional_jacobi_iteration(current_guess, rhses, lu_conv_kernel, d_inverse)
        for k in range(self.n_iterations-1):
            new_guess = self.convolutional_jacobi_iteration(new_guess, rhses, lu_conv_kernel, d_inverse)
            
        return new_guess

        

        
        
