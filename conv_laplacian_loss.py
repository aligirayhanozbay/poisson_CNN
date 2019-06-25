import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
import opt_einsum as oe
import numpy as np

class convolutional_poisson_loss():
    def __init__(self, h = 0.05, compressible = False, stencil = 'five-point', return_pointwise = False, data_format = 'channels_first'):
        self.return_pointwise = return_pointwise
        self.compressible = compressible
        self.data_format = data_format
        inp_soln = Input((1, None, None))
        inp_dx = Input((1,))
        if compressible == False:
            if stencil == 'nine-point':
                w = np.zeros((5,5,1,1), dtype = tf.keras.backend.floatx())
                w[2,0,0,0] = -1
                w[0,2,0,0] = -1
                w[1,2,0,0] = 16.0
                w[2,1,0,0] = 16.0
                w[2,2,0,0] = -60
                w[3,2,0,0] = 16.0
                w[2,3,0,0] = 16.0
                w[2,4,0,0] = -1
                w[4,2,0,0] = -1
                w = w/12.0
                kernel_size = 5
            elif stencil == 'five-point':
                w = np.zeros((3,3,1,1), dtype = tf.keras.backend.floatx())
                w[1,1,0,0] = -4.0
                w[0,1,0,0] = 1.0
                w[1,0,0,0] = 1.0
                w[2,1,0,0] = 1.0
                w[1,2,0,0] = 1.0
                kernel_size = 3
            laplacian_kernel = Conv2D(filters=1, kernel_size=kernel_size, activation='linear', data_format = data_format, padding = 'valid', use_bias = False)(inp_soln)
            def f(x):
                try:
                    return oe.contract('i...,i->i...', x[0], tf.squeeze(1/(x[1]**2), axis = 1))
                except:
                    return x[0]
            dx_adjustment = tf.keras.layers.Lambda(f)([laplacian_kernel, inp_dx])
            mod = Model([inp_soln, inp_dx],laplacian_kernel)
            mod.set_weights([tf.constant(w, dtype=tf.keras.backend.floatx())])
        else:
            if stencil == 'nine-point':
                w = np.zeros((5,5,1,2), dtype = tf.keras.backend.floatx())
                w[0,2,0,0] = 1.0
                w[1,2,0,0] = -8.0
                w[2,2,0,0] = 0.0
                w[3,2,0,0] = 8.0
                w[4,2,0,0] = -1.0
                w = w/12.0
                kernel_size = 5
            if stencil == 'five-point':
                w = np.zeros((3,3,1,2), dtype = tf.keras.backend.floatx())
                w[0,1,0,0] = -1.0
                w[1,1,0,0] = 0.0
                w[2,1,0,0] = 1.0
                w = w/2.0
                kernel_size = 3
            w[...,0,1] = w[...,0,0].transpose()
            inp_rho = Input((1, None, None))
            grad_u = Conv2D(filters = 2, kernel_size = kernel_size, activation = 'linear', data_format = data_format, padding = 'valid', use_bias = False)(inp_soln)
            def f(x):
                try:
                    return oe.contract('i...,i->i...', x[0], tf.squeeze(1/(2*x[1]), axis = 1), backend = 'tensorflow')
                except:
                    return x[0]
            dx_adjustment_1 = tf.keras.layers.Lambda(f)([grad_u, inp_dx])
            if data_format == 'channels_first':
                def g(x):
                    try:
                        return oe.contract('...,...->...', x[0], x[1][...,kernel_size//2:-kernel_size//2,kernel_size//2:-kernel_size//2], backend = 'tensorflow')
                    except:
                        return x[0]
                rho_adjustment = tf.keras.layers.Lambda(g)([dx_adjustment_1, inp_rho])
            else:
                def g(x):
                    try:
                        return oe.contract('...,...->...', x[0], x[1][:,kernel_size//2:-kernel_size//2,kernel_size//2:-kernel_size//2,:], backend = 'tensorflow')
                    except:
                        return x[0]
                rho_adjustment = tf.keras.layers.Lambda(g)([dx_adjustment_1, inp_rho])                                 
            div = Conv2D(filters = 1, kernel_size = kernel_size, activation = 'linear', data_format = data_format, padding = 'valid', use_bias = False)(rho_adjustment)
            mod = Model([inp_soln, inp_rho, inp_dx], div)
            mod.set_weights([w, w.transpose((0,1,3,2))])
        self.mod = mod
    def evaluate_loss(self, soln, rhs, dx, rho = None):
        if self.compressible and (rho is None):
            raise(ValueError('rho must be provided for the compressible case'))
        if (not self.compressible) and (rho is not None):
            print('Warning: rho has been provided for an incompressible instance, and will be disregarded.')

        if isinstance(dx, float):
            dx = dx * tf.ones((soln.shape[0],1), dtype = tf.keras.backend.floatx())

        if self.compressible:
            out = self.mod([soln, rho, dx])
        else:
            out = self.mod([soln, dx])

        if self.data_format == 'channels_first':
            out = (out - rhs[...,(rhs.shape[-2] - out.shape[-2])//2:-(rhs.shape[-2] - out.shape[-2])//2,(rhs.shape[-1] - out.shape[-1])//2:-(rhs.shape[-1] - out.shape[-1])//2])**2
        else:
            out = (out - rhs[:,(rhs.shape[-3] - out.shape[-3])//2:-(rhs.shape[-3] - out.shape[-3])//2,(rhs.shape[-2] - out.shape[-2])//2:-(rhs.shape[-2] - out.shape[-2])//2,:])**2
        if self.return_pointwise:
            return out
        else:
            return tf.reduce_mean(out)

# def conv_laplacian_loss(h, return_pointwise = False):
#     inp = Input((1,None, None))
#     laplacian_kernel = Conv2D(filters=1, kernel_size=5, activation='linear', data_format='channels_first', padding = 'same')(inp)
#     mod = Model(inp,laplacian_kernel)
#     w = np.zeros((5,5,1,1), dtype = np.float64)
#     w[2,0,0,0] = -1
#     w[0,2,0,0] = -1
#     w[1,2,0,0] = 16.0
#     w[2,1,0,0] = 16.0
#     w[2,2,0,0] = -60
#     w[3,2,0,0] = 16.0
#     w[2,3,0,0] = 16.0
#     w[2,4,0,0] = -1
#     w[4,2,0,0] = -1
#     mod.set_weights([(1/(12*h**2))*tf.constant(w, dtype=tf.float64),tf.constant([0.0], dtype=tf.float64)])
#     #pdb.set_trace()
#     if return_pointwise:
#         if int(tf.__version__[0]) < 2:
#             import tensorflow.contrib.eager as tfe
#             @tf.contrib.eager.defun
#             def laplacian_loss(rhs, solution):
#                 return (mod(solution)[:,:,2:-2,2:-2]-rhs[:,:,2:-2,2:-2])**2
#         else: 
#             @tf.function
#             def laplacian_loss(rhs, solution):
#                 return (mod(solution)[:,:,2:-2,2:-2]-rhs[:,:,2:-2,2:-2])**2
#     else:
#         if int(tf.__version__[0]) < 2:
#             @tf.contrib.eager.defun
#             def laplacian_loss(rhs, solution):
#                 return tf.reduce_sum((mod(solution)[:,:,2:-2,2:-2]-rhs[:,:,2:-2,2:-2])**2)/tf.cast(tf.reduce_prod(rhs[:,:,2:-2,2:-2].shape), rhs.dtype)
#         else:
#             @tf.function
#             def laplacian_loss(rhs, solution):
#                 return tf.reduce_sum((mod(solution)[:,:,2:-2,2:-2]-rhs[:,:,2:-2,2:-2])**2)/tf.cast(tf.reduce_prod(rhs[:,:,2:-2,2:-2].shape), rhs.dtype)
#     return laplacian_loss

# def compressible_poisson_loss(dx, dy = None, return_pointwise = False, data_format = data_format):
#     inp = Input((1,-1,-1))
