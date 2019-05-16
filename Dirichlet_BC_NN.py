import tensorflow as tf
from WeightedContractionLayer import WeightedContractionLayer
from Upsample import Upsample2
import itertools
import opt_einsum as oe
import numpy as np
from ABC import ABC, abstractmethod

class Model_With_Integral_Loss_ABC(ABC, tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def integral_loss(self, y_true, y_pred):
        try: #try except block to handle keras model init shenanigans
            if self.data_format == 'channels_first':
                c = 0.5* tf.concat([self.dx * int(y_true.shape[-2]), self.dx * int(y_true.shape[-1])],1)
                coords = np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-2]),np.linspace(-1, 1, y_true.shape[-1]),indexing = 'xy'), dtype = tf.keras.backend.floatx()).transpose((1,2,0)) #coordinates of each grid pt in the domain
            else:
                c = 0.5 * tf.concat([self.dx * int(y_true.shape[-3]), self.dx * int(y_true.shape[-2])],1)
                coords = np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-3]),np.linspace(-1, 1, y_true.shape[-2]),indexing = 'xy'), dtype = tf.keras.backend.floatx()).transpose((1,2,0)) #coordinates of each grid pt in the domain
        except:
            return 0.0*(y_true - y_pred)
        image_coords = [coords[0,:,0], coords[:,1,1]] #x and y coordinates separately
        quadrature_x, quadrature_w = tuple([x.astype(tf.keras.backend.floatx()) for x in np.polynomial.legendre.leggauss(self.n_quadpts)])
        #quadrature_x, quadrature_w = tuple([np.polynomial.legendre.leggauss(self.n_quadpts)[i].astype(np.float64) for i in range(2)])

        quadpts = tf.constant(np.array(np.meshgrid(quadrature_x,quadrature_x,indexing = 'xy')).transpose((1,2,0)),dtype = tf.keras.backend.floatx())
        #quadweights = tf.reduce_prod(c)*tf.tensordot(tf.squeeze(quadrature_w),tf.squeeze(quadrature_w),axes = 0)
        indices = [[],[]] #indices between each quadrature point lies - indices[0] is in x-dir and indices[1] is in the y-dir
        quad_coords = [quadpts[0,:,0], quadpts[:,1,1]] #x and y coordinates of each quad pt respectively
        #find the indices of coords between which every quad. pt. lies
        for i in range(len(indices)):
            j=0
            #does not work if more than 2 quad pts fall within 1 cell - fix later
            while len(indices[i]) < quadpts.shape[0] and j<image_coords[i].shape[0]:
                try:
                    if abs(float(quad_coords[i][len(indices[i])] - image_coords[i][j])) == float(min(abs(quad_coords[i][len(indices[i])] - image_coords[i][j-1]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j+1]))):
                        if quad_coords[i][len(indices[i])] - image_coords[i][j] < 0:
                            indices[i].append((j-1,j))
                        else:
                            indices[i].append((j,j+1))
                except:
                    if abs(float(quad_coords[i][len(indices[i])] - image_coords[i][j])) == float(min(abs(quad_coords[i][len(indices[i])] - image_coords[i][j-1]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j]))):
                        indices[i].append((j-1,j))
                j+=1
        
        index_combinations = tf.Variable(tf.zeros((quadpts.shape[0], quadpts.shape[1], 4 , 2), dtype = tf.int32), trainable = False, dtype = tf.int32) #array storing the 4 index combinations on the original grid which surround each quad. pt.
        corners = tf.Variable(tf.zeros((quadpts.shape[0], quadpts.shape[1], 2 , 2), dtype = np.int32), dtype = tf.int32, trainable = False) #array storing the lower left corner and the upper right corner of each box stored in index_combinations. effectively this will contain [[xmin,ymin],[xmax,ymax]] for the rectangle around each quad pt.
        s=tf.constant(indices)
        for i in range(self.n_quadpts):
            for j in range(self.n_quadpts):
                index_combinations[i,j,:,:].assign(np.array(list(itertools.product(np.array(s)[0,i,:],np.array(s)[1,j,:]))))
        for i in range(self.n_quadpts):
            for j in range(self.n_quadpts):
                corners[i,j,:,:].assign(tf.stack([s[0,i,:],s[1,j,:]]))
        corners = tf.transpose(corners,(0,1,3,2))
        corner_coords = tf.gather_nd(tf.transpose(coords,(1,0,2)),corners)
        
        #compute the coefficients [b_11,b_12,b_21,b_22]
        #steps:
        #1. compute transpose(invert(array([[1,xmin,ymin,xmin*ymin],[1,xmin,ymax,xmin*ymax],[1,xmax,ymin,xmax*ymin],[1,xmax,ymax,xmax*ymax]]))) for the rectangle around each quad pt.
        #2. compute array([1,x_quadpt, y_quadpt, x_quadpt*y_quadpt]) for each quadpt
        #3. multiply the result of 1 and 2 for each quad pt.
        interpolation_matrix = tf.Variable(tf.ones((self.n_quadpts,self.n_quadpts,4,4), dtype = tf.keras.backend.floatx()), dtype = tf.keras.backend.floatx())
        interpolation_matrix[:,:,0:2,1].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,0:2,1],corner_coords[:,:,0,0], backend = 'tensorflow'))
        interpolation_matrix[:,:,2:,1].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,2:,1],corner_coords[:,:,1,0], backend = 'tensorflow'))
        interpolation_matrix[:,:,0::2,2].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,0::2,2],corner_coords[:,:,0,1], backend = 'tensorflow'))
        interpolation_matrix[:,:,1::2,2].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,1::2,2], corner_coords[:,:,1,1], backend = 'tensorflow'))
        interpolation_matrix[:,:,:,3].assign(oe.contract('...,...,...->...',interpolation_matrix[:,:,:,3],interpolation_matrix[:,:,:,1], interpolation_matrix[:,:,:,2], backend = 'tensorflow'))
        interpolation_matrix = tf.transpose(tf.linalg.inv(interpolation_matrix), (0,1,3,2))
        q = tf.Variable(tf.ones((self.n_quadpts,self.n_quadpts,4), dtype = tf.keras.backend.floatx()), dtype = tf.keras.backend.floatx())
        q[:,:,1].assign(tf.transpose(quadpts[:,:,0]))
        q[:,:,2].assign(tf.transpose(quadpts[:,:,1]))
        q[:,:,3].assign(tf.multiply(q[:,:,1],q[:,:,2]))
        
        b = oe.contract('ijkl, ijl->ijk', interpolation_matrix, q, backend = 'tensorflow')
        
        quadweights = oe.contract('i,j,k->ijk',tf.squeeze(quadrature_w),tf.squeeze(quadrature_w), tf.reduce_prod(c,axis = 1), backend = 'tensorflow')
        if self.data_format == 'channels_first':
            interp_pts = tf.squeeze(tf.gather_nd(tf.transpose(y_true - y_pred, (2,3,1,0)), index_combinations), axis = 3)
        else:
            interp_pts = tf.squeeze(tf.gather_nd(tf.transpose(y_true - y_pred, (1,2,3,0)), index_combinations), axis = 3)

        values_at_quad_pts = oe.contract('ijkl, ijk->ijl', interp_pts, b, backend = 'tensorflow')

        loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(quadweights, values_at_quad_pts**self.p), axis = (0,1))**(1/self.p))

        if self.mae_component_weight != 0.0:
            loss = loss + self.mae_component_weight * tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))
        if self.mse_component_weight != 0.0:
            loss = loss + self.mse_component_weight * tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        return loss

class SepConvBlock(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', separable_kernel_size  = (5,256), nonsep_kernel_size = 5, separable_activation = tf.nn.leaky_relu, nonsep_activation = tf.nn.leaky_relu, separable_filters = 8, nonsep_filters = 4):
        super().__init__()
        self.separableconv2d = tf.keras.layers.SeparableConv2D(separable_filters, kernel_size = separable_kernel_size, padding = 'same', activation = separable_activation, data_format = data_format)
        self.conv2d = tf.keras.layers.Conv2D(filters = nonsep_filters, kernel_size = nonsep_kernel_size, activation = nonsep_activation, padding = 'same', data_format = data_format)
        
    def call(self, inp):
        return self.conv2d(self.separableconv2d(inp))

#best candidate
class Dirichlet_BC_NN_2B(tf.keras.models.Model): #variant to include dx info
    def __init__(self, data_format = 'channels_first', x_output_resolution = 256, n_sepconvblocks = 3, mae_component_weight = 0.0, mse_component_weight = 0.0, n_quadpts= 20, Lp_norm_power = 2):
        super().__init__()
        self.mae_component_weight = mae_component_weight
        self.mse_component_weight = mse_component_weight
        self.n_quadpts = n_quadpts
        self.p = Lp_norm_power
        self.x_output_resolution = x_output_resolution
        self.data_format = data_format
        self.conv1d_0 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        if data_format == 'channels_first':
            sepconvkernelshape = (256,5)
        else:
            sepconvkernelshape = (5,256)
        self.sepconvblocks_3 = [SepConvBlock(data_format = data_format, separable_kernel_size = sepconvkernelshape, separable_filters = 10, nonsep_filters = 10) for i in range(n_sepconvblocks)]
        
        self.conv2d_4 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        
        self.output_upsample_4 = Upsample2([-1, -1], data_format = data_format , resize_method = tf.image.ResizeMethod.BICUBIC)
        
        self.conv2d_5 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 9, activation = tf.tanh, padding = 'same', data_format = data_format)
        
        self.dx_dense_0 = tf.keras.layers.Dense(4, activation = tf.nn.relu)
        self.dx_dense_1 = tf.keras.layers.Dense(4, activation = tf.nn.relu)
        self.dx_dense_2 = tf.keras.layers.Dense(8, activation = tf.nn.softmax)
    def call(self, inputs):
        self.dx = inputs[1]
        dx_res = 1/(1e-8 + 10 * self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(inputs[1]))))
        
        if self.data_format == 'channels_first':
            self.input_length = inputs[0].shape[-1]
            contr_expr = 'ijk,ij->ijk'
        else:
            self.input_length = inputs[0].shape[-2]
            contr_expr = 'ikj,ij->ikj'
        out = self.conv1d_2(tf.einsum(contr_expr, self.conv1d_1(self.conv1d_0(inputs[0])), dx_res))
        
        if self.data_format == 'channels_first':
            out = tf.expand_dims(out, axis = 1)
            newshape = [self.x_output_resolution, self.input_length]
        else:
            out = tf.expand_dims(out, axis = 3)
            newshape = [self.input_length, self.x_output_resolution]
            
        for scb in self.sepconvblocks_3:
            out = scb(out)
            
        out = self.output_upsample_4([self.conv2d_4(out), newshape])
        return self.conv2d_5(out)
    
    def integral_loss(self, y_true, y_pred):
        try:
            if self.data_format == 'channels_first':
                c = 0.5* tf.concat([self.dx * int(y_true.shape[-2]), self.dx * int(y_true.shape[-1])],1)
                coords = np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-2]),np.linspace(-1, 1, y_true.shape[-1]),indexing = 'xy'), dtype = np.float64).transpose((1,2,0)) #coordinates of each grid pt in the domain
            else:
                c = 0.5 * tf.concat([self.dx * int(y_true.shape[-3]), self.dx * int(y_true.shape[-2])],1)
                coords = np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-3]),np.linspace(-1, 1, y_true.shape[-2]),indexing = 'xy'), dtype = np.float64).transpose((1,2,0)) #coordinates of each grid pt in the domain
        except:
            return 0.0*(y_true - y_pred)
        image_coords = [coords[0,:,0], coords[:,1,1]] #x and y coordinates separately
        quadrature_x, quadrature_w = tuple([np.polynomial.legendre.leggauss(self.n_quadpts)[i].astype(np.float64) for i in range(2)])

        quadpts = tf.constant(np.array(np.meshgrid(quadrature_x,quadrature_x,indexing = 'xy')).transpose((1,2,0)),dtype = tf.float64)
        #quadweights = tf.reduce_prod(c)*tf.tensordot(tf.squeeze(quadrature_w),tf.squeeze(quadrature_w),axes = 0)
        indices = [[],[]] #indices between each quadrature point lies - indices[0] is in x-dir and indices[1] is in the y-dir
        quad_coords = [quadpts[0,:,0], quadpts[:,1,1]] #x and y coordinates of each quad pt respectively
        #find the indices of coords between which every quad. pt. lies
        for i in range(len(indices)):
            j=0
            #does not work if more than 2 quad pts fall within 1 cell - fix later
            while len(indices[i]) < quadpts.shape[0] and j<image_coords[i].shape[0]:
                try:
                    if abs(float(quad_coords[i][len(indices[i])] - image_coords[i][j])) == float(min(abs(quad_coords[i][len(indices[i])] - image_coords[i][j-1]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j+1]))):
                        if quad_coords[i][len(indices[i])] - image_coords[i][j] < 0:
                            indices[i].append((j-1,j))
                        else:
                            indices[i].append((j,j+1))
                except:
                    if abs(float(quad_coords[i][len(indices[i])] - image_coords[i][j])) == float(min(abs(quad_coords[i][len(indices[i])] - image_coords[i][j-1]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j]))):
                        indices[i].append((j-1,j))
                j+=1
        
        index_combinations = tf.Variable(tf.zeros((quadpts.shape[0], quadpts.shape[1], 4 , 2), dtype = tf.int32), trainable = False, dtype = tf.int32) #array storing the 4 index combinations on the original grid which surround each quad. pt.
        corners = tf.Variable(tf.zeros((quadpts.shape[0], quadpts.shape[1], 2 , 2), dtype = np.int32), dtype = tf.int32, trainable = False) #array storing the lower left corner and the upper right corner of each box stored in index_combinations. effectively this will contain [[xmin,ymin],[xmax,ymax]] for the rectangle around each quad pt.
        s=tf.constant(indices)
        for i in range(self.n_quadpts):
            for j in range(self.n_quadpts):
                index_combinations[i,j,:,:].assign(np.array(list(itertools.product(np.array(s)[0,i,:],np.array(s)[1,j,:]))))
        for i in range(self.n_quadpts):
            for j in range(self.n_quadpts):
                corners[i,j,:,:].assign(tf.stack([s[0,i,:],s[1,j,:]]))
        corners = tf.transpose(corners,(0,1,3,2))
        corner_coords = tf.gather_nd(tf.transpose(coords,(1,0,2)),corners)
        
        #compute the coefficients [b_11,b_12,b_21,b_22]
        #steps:
        #1. compute transpose(invert(array([[1,xmin,ymin,xmin*ymin],[1,xmin,ymax,xmin*ymax],[1,xmax,ymin,xmax*ymin],[1,xmax,ymax,xmax*ymax]]))) for the rectangle around each quad pt.
        #2. compute array([1,x_quadpt, y_quadpt, x_quadpt*y_quadpt]) for each quadpt
        #3. multiply the result of 1 and 2 for each quad pt.
        interpolation_matrix = tf.Variable(tf.ones((self.n_quadpts,self.n_quadpts,4,4), dtype = tf.keras.backend.floatx()), dtype = tf.keras.backend.floatx())
        interpolation_matrix[:,:,0:2,1].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,0:2,1],corner_coords[:,:,0,0], backend = 'tensorflow'))
        interpolation_matrix[:,:,2:,1].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,2:,1],corner_coords[:,:,1,0], backend = 'tensorflow'))
        interpolation_matrix[:,:,0::2,2].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,0::2,2],corner_coords[:,:,0,1], backend = 'tensorflow'))
        interpolation_matrix[:,:,1::2,2].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,1::2,2], corner_coords[:,:,1,1], backend = 'tensorflow'))
        interpolation_matrix[:,:,:,3].assign(oe.contract('...,...,...->...',interpolation_matrix[:,:,:,3],interpolation_matrix[:,:,:,1], interpolation_matrix[:,:,:,2], backend = 'tensorflow'))
        interpolation_matrix = tf.transpose(tf.linalg.inv(interpolation_matrix), (0,1,3,2))
        q = tf.Variable(tf.ones((self.n_quadpts,self.n_quadpts,4), dtype = tf.keras.backend.floatx()), dtype = tf.keras.backend.floatx())
        q[:,:,1].assign(tf.transpose(quadpts[:,:,0]))
        q[:,:,2].assign(tf.transpose(quadpts[:,:,1]))
        q[:,:,3].assign(tf.multiply(q[:,:,1],q[:,:,2]))
        
        b = oe.contract('ijkl, ijl->ijk', interpolation_matrix, q, backend = 'tensorflow')


        quadweights = oe.contract('i,j,k->ijk',tf.squeeze(quadrature_w),tf.squeeze(quadrature_w), tf.reduce_prod(c,axis = 1), backend = 'tensorflow')
        if self.data_format == 'channels_first':
            interp_pts = tf.squeeze(tf.gather_nd(tf.transpose(y_true - y_pred, (2,3,1,0)), index_combinations), axis = 3)
        else:
            interp_pts = tf.squeeze(tf.gather_nd(tf.transpose(y_true - y_pred, (1,2,3,0)), index_combinations), axis = 3)

        values_at_quad_pts = oe.contract('ijkl, ijk->ijl', interp_pts, b, backend = 'tensorflow')

        loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(quadweights, values_at_quad_pts**self.p), axis = (0,1))**(1/self.p))

        if self.mae_component_weight != 0.0:
            loss = loss + self.mae_component_weight * tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))
        if self.mse_component_weight != 0.0:
            loss = loss + self.mse_component_weight * tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        return loss

#best candidate
class Dirichlet_BC_NN_2C(tf.keras.models.Model): #variant to include dx AND Lx/Ly info
    def __init__(self, data_format = 'channels_first', x_output_resolution = 256, n_sepconvblocks = 3, mae_component_weight = 0.0, mse_component_weight = 0.0, n_quadpts= 20, Lp_norm_power = 2):
        super().__init__()
        self.mae_component_weight = mae_component_weight
        self.mse_component_weight = mse_component_weight
        self.n_quadpts = n_quadpts
        self.p = Lp_norm_power
        self.x_output_resolution = x_output_resolution
        self.data_format = data_format
        self.conv1d_0 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=64, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        if data_format == 'channels_first':
            sepconvkernelshape = (256,5)
        else:
            sepconvkernelshape = (5,256)
        self.sepconvblocks_3 = [SepConvBlock(data_format = data_format, separable_kernel_size = sepconvkernelshape, separable_filters = 10, nonsep_filters = 10) for i in range(n_sepconvblocks+1)]
        
        self.conv2d_4_0 = tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        self.conv2d_4_1 = tf.keras.layers.Conv2D(filters = 12, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        self.conv2d_4_2 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        
        
        self.output_upsample_4 = Upsample2([-1, -1], data_format = data_format , resize_method = tf.image.ResizeMethod.BICUBIC)
        
        self.conv2d_5_0 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 9, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        
        self.conv2d_5_1 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 9, activation = tf.nn.tanh, padding = 'same', data_format = data_format)
        
        
        self.dx_dense_0 = tf.keras.layers.Dense(8, activation = tf.nn.leaky_relu)
        self.dx_dense_1 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dx_dense_2 = tf.keras.layers.Dense(256, activation = tf.nn.tanh)
    def call(self, inputs):
        self.x_output_resolution = inputs[2]
        
        self.dx = inputs[1]
        try: #stupid hack 1 to get past keras '?' tensor dimensions via try except block
            if self.data_format == 'channels_first':
                self.domain_info = oe.contract('ij,j->ij',tf.tile(inputs[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[2]), tf.keras.backend.floatx())], axis = 0), backend = 'tensorflow')
                self.input_length = inputs[0].shape[-1]
                contr_expr = 'ikl,ik->ikl'
            else:
                self.domain_info = oe.contract('ij,j->ij',tf.tile(inputs[1], [1,3]), tf.stack([tf.constant(1.0, dtype = tf.keras.backend.floatx()), tf.cast(self.x_output_resolution, tf.keras.backend.floatx()), tf.cast(tf.constant(inputs[0].shape[1]), tf.keras.backend.floatx())], axis = 0), backend = 'tensorflow')
                self.input_length = inputs[0].shape[-2]
                contr_expr = 'ilk,ik->ilk'
        except:
            if self.data_format == 'channels_first':
                self.domain_info = tf.tile(inputs[1], [1,3])
                self.input_length = inputs[0].shape[-1]
                contr_expr = 'ikl,ik->ikl'
            else:
                self.domain_info = tf.tile(inputs[1], [1,3])
                self.input_length = inputs[0].shape[-2]
                contr_expr = 'ilk,ik->ilk'
        dx_res = self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(self.domain_info)))

        out = tf.einsum(contr_expr, self.conv1d_2(self.conv1d_1(self.conv1d_0(inputs[0]))), dx_res)
        try: #stupid hack 2 to get past keras '?' tensor dims via try except block
            if self.data_format == 'channels_first':
                out = tf.expand_dims(out, axis = 1)
                newshape = [int(self.x_output_resolution), self.input_length]
            else:
                out = tf.expand_dims(out, axis = 3)
                newshape = [self.input_length, int(self.x_output_resolution)]
            
            for scb in self.sepconvblocks_3:
                out = scb(out)

            out = self.output_upsample_4([self.conv2d_4_2(self.conv2d_4_1(self.conv2d_4_0(out))), newshape])
            return self.conv2d_5_1(self.conv2d_5_0(out))
        except:
            if self.data_format == 'channels_first': # 
                #out = tf.expand_dims(out, axis = 1)
                newshape = [64, self.input_length]
            else:
                #out = tf.expand_dims(out, axis = 3)
                newshape = [self.input_length, 64]
            for scb in self.sepconvblocks_3:
                out = scb(out)
            
            out = self.output_upsample_4([self.conv2d_4_2(self.conv2d_4_1(self.conv2d_4_0(out))), newshape])
            return self.conv2d_5_1(self.conv2d_5_0(out))

    
    def integral_loss(self, y_true, y_pred):
        try:
            if self.data_format == 'channels_first':
                c = 0.5* tf.concat([self.dx * int(y_true.shape[-2]), self.dx * int(y_true.shape[-1])],1)
                coords = np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-2]),np.linspace(-1, 1, y_true.shape[-1]),indexing = 'xy'), dtype = tf.keras.backend.floatx()).transpose((1,2,0)) #coordinates of each grid pt in the domain
            else:
                c = 0.5 * tf.concat([self.dx * int(y_true.shape[-3]), self.dx * int(y_true.shape[-2])],1)
                coords = np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-3]),np.linspace(-1, 1, y_true.shape[-2]),indexing = 'xy'), dtype = tf.keras.backend.floatx()).transpose((1,2,0)) #coordinates of each grid pt in the domain
        except:
            return 0.0*(y_true - y_pred)
        image_coords = [coords[0,:,0], coords[:,1,1]] #x and y coordinates separately
        quadrature_x, quadrature_w = tuple([x.astype(tf.keras.backend.floatx()) for x in np.polynomial.legendre.leggauss(self.n_quadpts)])
        #quadrature_x, quadrature_w = tuple([np.polynomial.legendre.leggauss(self.n_quadpts)[i].astype(np.float64) for i in range(2)])

        quadpts = tf.constant(np.array(np.meshgrid(quadrature_x,quadrature_x,indexing = 'xy')).transpose((1,2,0)),dtype = tf.keras.backend.floatx())
        #quadweights = tf.reduce_prod(c)*tf.tensordot(tf.squeeze(quadrature_w),tf.squeeze(quadrature_w),axes = 0)
        indices = [[],[]] #indices between each quadrature point lies - indices[0] is in x-dir and indices[1] is in the y-dir
        quad_coords = [quadpts[0,:,0], quadpts[:,1,1]] #x and y coordinates of each quad pt respectively
        #find the indices of coords between which every quad. pt. lies
        for i in range(len(indices)):
            j=0
            #does not work if more than 2 quad pts fall within 1 cell - fix later
            while len(indices[i]) < quadpts.shape[0] and j<image_coords[i].shape[0]:
                try:
                    if abs(float(quad_coords[i][len(indices[i])] - image_coords[i][j])) == float(min(abs(quad_coords[i][len(indices[i])] - image_coords[i][j-1]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j+1]))):
                        if quad_coords[i][len(indices[i])] - image_coords[i][j] < 0:
                            indices[i].append((j-1,j))
                        else:
                            indices[i].append((j,j+1))
                except:
                    if abs(float(quad_coords[i][len(indices[i])] - image_coords[i][j])) == float(min(abs(quad_coords[i][len(indices[i])] - image_coords[i][j-1]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j]))):
                        indices[i].append((j-1,j))
                j+=1
        
        index_combinations = tf.Variable(tf.zeros((quadpts.shape[0], quadpts.shape[1], 4 , 2), dtype = tf.int32), trainable = False, dtype = tf.int32) #array storing the 4 index combinations on the original grid which surround each quad. pt.
        corners = tf.Variable(tf.zeros((quadpts.shape[0], quadpts.shape[1], 2 , 2), dtype = np.int32), dtype = tf.int32, trainable = False) #array storing the lower left corner and the upper right corner of each box stored in index_combinations. effectively this will contain [[xmin,ymin],[xmax,ymax]] for the rectangle around each quad pt.
        s=tf.constant(indices)
        for i in range(self.n_quadpts):
            for j in range(self.n_quadpts):
                index_combinations[i,j,:,:].assign(np.array(list(itertools.product(np.array(s)[0,i,:],np.array(s)[1,j,:]))))
        for i in range(self.n_quadpts):
            for j in range(self.n_quadpts):
                corners[i,j,:,:].assign(tf.stack([s[0,i,:],s[1,j,:]]))
        corners = tf.transpose(corners,(0,1,3,2))
        corner_coords = tf.gather_nd(tf.transpose(coords,(1,0,2)),corners)
        
        #compute the coefficients [b_11,b_12,b_21,b_22]
        #steps:
        #1. compute transpose(invert(array([[1,xmin,ymin,xmin*ymin],[1,xmin,ymax,xmin*ymax],[1,xmax,ymin,xmax*ymin],[1,xmax,ymax,xmax*ymax]]))) for the rectangle around each quad pt.
        #2. compute array([1,x_quadpt, y_quadpt, x_quadpt*y_quadpt]) for each quadpt
        #3. multiply the result of 1 and 2 for each quad pt.
        interpolation_matrix = tf.Variable(tf.ones((self.n_quadpts,self.n_quadpts,4,4), dtype = tf.keras.backend.floatx()), dtype = tf.keras.backend.floatx())
        interpolation_matrix[:,:,0:2,1].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,0:2,1],corner_coords[:,:,0,0], backend = 'tensorflow'))
        interpolation_matrix[:,:,2:,1].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,2:,1],corner_coords[:,:,1,0], backend = 'tensorflow'))
        interpolation_matrix[:,:,0::2,2].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,0::2,2],corner_coords[:,:,0,1], backend = 'tensorflow'))
        interpolation_matrix[:,:,1::2,2].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,1::2,2], corner_coords[:,:,1,1], backend = 'tensorflow'))
        interpolation_matrix[:,:,:,3].assign(oe.contract('...,...,...->...',interpolation_matrix[:,:,:,3],interpolation_matrix[:,:,:,1], interpolation_matrix[:,:,:,2], backend = 'tensorflow'))
        interpolation_matrix = tf.transpose(tf.linalg.inv(interpolation_matrix), (0,1,3,2))
        q = tf.Variable(tf.ones((self.n_quadpts,self.n_quadpts,4), dtype = tf.keras.backend.floatx()), dtype = tf.keras.backend.floatx())
        q[:,:,1].assign(tf.transpose(quadpts[:,:,0]))
        q[:,:,2].assign(tf.transpose(quadpts[:,:,1]))
        q[:,:,3].assign(tf.multiply(q[:,:,1],q[:,:,2]))
        
        b = oe.contract('ijkl, ijl->ijk', interpolation_matrix, q, backend = 'tensorflow')
        
        quadweights = oe.contract('i,j,k->ijk',tf.squeeze(quadrature_w),tf.squeeze(quadrature_w), tf.reduce_prod(c,axis = 1), backend = 'tensorflow')
        if self.data_format == 'channels_first':
            interp_pts = tf.squeeze(tf.gather_nd(tf.transpose(y_true - y_pred, (2,3,1,0)), index_combinations), axis = 3)
        else:
            interp_pts = tf.squeeze(tf.gather_nd(tf.transpose(y_true - y_pred, (1,2,3,0)), index_combinations), axis = 3)

        values_at_quad_pts = oe.contract('ijkl, ijk->ijl', interp_pts, b, backend = 'tensorflow')

        loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(quadweights, values_at_quad_pts**self.p), axis = (0,1))**(1/self.p))

        if self.mae_component_weight != 0.0:
            loss = loss + self.mae_component_weight * tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))
        if self.mse_component_weight != 0.0:
            loss = loss + self.mse_component_weight * tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        return loss

