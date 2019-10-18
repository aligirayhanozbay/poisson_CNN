import tensorflow as tf
import numpy as np
import opt_einsum as oe
import copy

from ..layers import SpatialPyramidPool
from .custom_blocks import ResnetBlock
from .Model_With_Integral_Loss import Model_With_Integral_Loss_ABC
from .Poisson_CNN import channels_first_flip_left_right

from ..dataset.generators.numerical import set_max_magnitude_in_batch as smmib
from ..misc import convolutional_poisson_loss

class Dirichlet_BC_NN_Series(Model_With_Integral_Loss_ABC):
    def __init__(self, data_format = 'channels_first', spp_levels = [[2],3,4,6,8,12,20,50], nmodes = 32, npolynomial = 5, conv_stages = 5, pooling_block_number = 5, final_channels = 10, initial_kernel_size = 19, final_kernel_size = 3, dense_layer_number = 5, initial_dense_layer_units = 250, kernel_regularizer = None, bias_regularizer = None, x_output_resolution = 256, convolutional_poisson_loss_stencil = 'five-point', **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format
        self.x_output_resolution = x_output_resolution
        self.nmodes = nmodes
        self.npolynomial = npolynomial
        self.modes_arange = tf.cast(tf.range(1,self.nmodes+1,delta=1), tf.keras.backend.floatx())
        
        if self.data_format == 'channels_first':
            self.space_dims_start = 2
        else:
            self.space_dims_start = 1

        
        spp_levels.append(None)
        self.conv0 = tf.keras.layers.Conv1D(filters = 2, kernel_size = initial_kernel_size, activation = tf.nn.leaky_relu, data_format = self.data_format, padding = 'same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer)
        self.pooling_blocks = []
        for j in range(pooling_block_number):
            conv_blocks = []
            for k in range(conv_stages):
                ksize = initial_kernel_size + (k*(final_kernel_size - initial_kernel_size)//(conv_stages-1))
                filters = final_channels-((k*final_channels)//(conv_stages))
                block = []
                if (k == 0) and (j != 0):
                    block.append(tf.keras.layers.AveragePooling1D(pool_size = 2**j, data_format = self.data_format))
                block.append(tf.keras.layers.Conv1D(filters = filters, kernel_size = ksize, activation=tf.nn.leaky_relu, data_format = self.data_format, padding = 'same', kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
                block.append(ResnetBlock(ndims = 1, data_format = self.data_format, filters = filters, kernel_size = ksize, activation=tf.nn.leaky_relu, kernel_regularizer = kernel_regularizer, bias_regularizer = bias_regularizer))
                if k == (conv_stages-1):
                    block.append(SpatialPyramidPool(spp_levels[:-(j+1)], data_format = self.data_format))
                conv_blocks.append(block)
            self.pooling_blocks.append(conv_blocks)

        self.dense_layers = []
        for k in range(dense_layer_number):
            units = initial_dense_layer_units + (k*(nmodes - initial_dense_layer_units)//(dense_layer_number-1))
            self.dense_layers.append(tf.keras.layers.Dense(units, activation = tf.nn.leaky_relu))
        #self.dense_layers.append(tf.keras.layers.Dense(nmodes+npolynomial*(npolynomial-1)/2, activation = tf.nn.leaky_relu))
        self.fourier_dense = tf.keras.layers.Dense(self.nmodes, activation = tf.nn.tanh) #26 sep  set activations to tanh
        if npolynomial != 0:
            self.poly_dense = tf.keras.layers.Dense(self.nmodes*self.npolynomial*(self.npolynomial+1)/2, activation = tf.nn.tanh)

    @staticmethod
    def build_polynomials(nx, order):
        order = order + 1
        result = []
        x = tf.cast(tf.keras.backend.arange(0,stop=nx)/(nx-1), tf.keras.backend.floatx())
        for k in range(order):
            if k == 0:
                result.append(tf.ones((nx), dtype = tf.keras.backend.floatx()))
            else:
                for j in range(k+1):
                    result.append((x**j)*((1-x)**(k-j)))
        return tf.stack(result)
            
            
    def call(self, inp):
        self.dx = inp[1]

        #set output x resolution
        try:
            if inp[2].shape[0] == 1: #set output shape
                self.x_output_resolution = int(inp[2])
            elif self.data_format == 'channels_first':
                self.x_output_resolution = int(inp[2][2])
            else:
                self.x_output_resolution = int(inp[2][1])
        except:
            pass

        #get domain info
        if self.data_format == 'channels_first': 
           domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[2]), tf.keras.backend.floatx())], axis = 0)) #calculate domain info
           input_length = inp[0].shape[2] #get mesh size in the direction of the provided BC
        else:
           domain_info = tf.einsum('ij,j->ij',tf.tile(inp[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inp[0].shape[1]), tf.keras.backend.floatx())], axis = 0))
           input_length = inp[0].shape[1]

        #actual NN bit
        amplitudes = self.conv0(inp[0])
        pooling_block_results = []
        for pooling_block in self.pooling_blocks:
            pbr = amplitudes + 0.0
            for block in pooling_block:
                for layer in block:
                    pbr = layer(pbr)
            pooling_block_results.append(pbr)
        amplitudes = tf.keras.backend.concatenate(pooling_block_results + [domain_info], 1)

        #amplitudes = self.spp(amplitudes)
        #amplitudes = tf.keras.backend.concatenate([amplitudes,domain_info], 1)

        for layer in self.dense_layers:
            amplitudes = layer(amplitudes)

        if self.npolynomial != 0:
            poly_amplitudes = tf.keras.backend.reshape(self.poly_dense(amplitudes), (-1, self.nmodes, int(self.npolynomial*(self.npolynomial+1)/2)))
        amplitudes = self.fourier_dense(amplitudes)

        mode_coefficient_adjustment = tf.einsum('i,ij->ij',(2/domain_info[:,2])**2,1/tf.sinh(tf.einsum('i,j->ij',domain_info[:,1]/domain_info[:,2],-self.modes_arange*np.pi))) #26 sep - squared 2/domain_info[:,2]
        
        amplitudes = amplitudes * mode_coefficient_adjustment
        

        #build grids
        sinh = tf.sinh(tf.einsum('i,j,k->ijk', domain_info[:,1]/domain_info[:,2], self.modes_arange, -tf.keras.backend.arange(0,self.x_output_resolution, dtype = tf.keras.backend.floatx())/tf.cast(self.x_output_resolution-1, tf.keras.backend.floatx())*tf.constant(np.pi, dtype = tf.keras.backend.floatx())))
        sin = tf.sin(tf.einsum('j,l->jl',  self.modes_arange, tf.keras.backend.arange(0,input_length, dtype = tf.keras.backend.floatx())/tf.cast(input_length-1, tf.keras.backend.floatx()))*tf.constant(np.pi, dtype = tf.keras.backend.floatx()))

        if self.npolynomial != 0:
            poly_amplitudes = tf.einsum('ijm,ij->ijm', poly_amplitudes, amplitudes) #26 sep - changed 2nd einsum argument from mode coeff adjustment to amplitudes
            if self.data_format == 'channels_first':
                poly = tf.einsum('ijm,ml->ijl', poly_amplitudes, self.build_polynomials(tf.keras.backend.shape(inp[0])[-1], self.npolynomial-1))
            else:
                poly = tf.einsum('ijm,ml->ijl', poly_amplitudes, self.build_polynomials(tf.keras.backend.shape(inp[0])[1], self.npolynomial-1))
        else:
            poly = tf.keras.backend.zeros(list(tf.shape(amplitudes)) + [tf.shape(sin)[-1]], dtype = tf.keras.backend.floatx())
        
        #calculate result
        try:
            #out = oe.contract('ijl,ijk->ikl', poly, sinh, backend='tensorflow')
            out = oe.contract('ijl,ijk->ikl',oe.contract('ij,jl->ijl', amplitudes, sin, backend = 'tensorflow')+poly,sinh, backend='tensorflow')
            #out = oe.contract('ijk,jl,ij->ikl', sinh, sin, amplitudes, backend = 'tensorflow')
        except:
            #out = tf.einsum('ijl,ijk->ikl', poly, sinh)
            out = tf.einsum('ijl,ijk->ikl',tf.einsum('ij,jl->ijl', amplitudes, sin)+poly,sinh)
            #out = tf.einsum('ijk,jl,ij->ikl', sinh, sin, amplitudes)
            
        
        if self.data_format == 'channels_first':
            out = tf.expand_dims(out, axis = 1)
            return smmib(channels_first_flip_left_right(out),1.0)
        else:
            out = tf.expand_dims(out, axis = -1)

    
        

        
        
        

        
