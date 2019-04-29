import tensorflow as tf
import numpy as np
from MergeWithAttention import MergeWithAttention2
from Upsample import Upsample2
from Lp_integral_norm import Lp_integral_norm
import itertools
import opt_einsum as oe

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
    
    
    
    
class Homogeneous_Poisson_NN_2(tf.keras.models.Model): #variant to include dx info
    def __init__(self, pooling_block_number = 6, resize_methods = None, data_format = 'channels_first', n_quadpts = 20, Lp_norm_power = 2, mae_component_weight = 0.0, mse_component_weight = 0.0):
        super().__init__()
        self.mae_component_weight = mae_component_weight
        self.mse_component_weight = mse_component_weight
        self.n_quadpts = n_quadpts
        self.p = Lp_norm_power
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

    #@tf.function
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

        # print('---data---')
        # print((y_true - y_pred).shape)
        # print('---index combinations---')
        # print(index_combinations.shape)
        # print('---interp_pts---')
        # print(interp_pts.shape)
        # print('---quadweights---')
        # print(quadweights[...,0])
        # print('---b---')
        # print(b.shape)

        values_at_quad_pts = oe.contract('ijkl, ijk->ijl', interp_pts, b, backend = 'tensorflow')

        loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(quadweights, values_at_quad_pts**self.p), axis = (0,1))**(1/self.p))
        # print('---values_at_quad_pts---')
        # print(values_at_quad_pts.shape)
        if self.mae_component_weight != 0.0:
            loss = loss + self.mae_component_weight * tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))
        if self.mse_component_weight != 0.0:
            loss = loss + self.mse_component_weight * tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        return loss

        
    
#    def integral_loss(self, y_true, y_pred):
#        losses = [Lp_integral_norm(y_true.shape[-2:], [0, self.dx[i,0]*y_true.shape[-2],0, self.dx[i,0]*y_true.shape[-1]], n_quadpts = 20, mse_component_weight = 1e+1) for i in range(y_true.shape[0])]
#        losses = [losses[i](tf.expand_dims(y_true[i], axis = 0), tf.expand_dims(y_pred[i], axis = 0)) for i in range(y_true.shape[0])]
#        return tf.reduce_mean(losses)
