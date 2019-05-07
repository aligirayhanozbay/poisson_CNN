import tensorflow as tf
import numpy as np
import opt_einsum as oe
import itertools
from Homogeneous_Poisson_NN import Homogeneous_Poisson_NN_2
from Dirichlet_BC_NN import Dirichlet_BC_NN_2B
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

class Poisson_CNN(tf.keras.models.Model):
    def __init__(self, bc_nn_parameters = {}, homogeneous_poisson_nn_parameters = {}, bc_nn_weights = None,  homogeneous_poisson_nn_weights = None, bc_nn_trainable = True, homogeneous_poisson_nn_trainable = True, data_format = 'channels_first', mae_component_weight = 0.0, mse_component_weight = 0.0, n_quadpts = 20, Lp_norm_power = 2, **kwargs):
        super().__init__(**kwargs)
        self.mae_component_weight = mae_component_weight
        self.mse_component_weight = mse_component_weight
        self.n_quadpts = n_quadpts
        self.p = Lp_norm_power
        
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
        
        self.bc_nn = Dirichlet_BC_NN_2B(**bc_nn_parameters)
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
#         laplace_soln = self.post_bc_nn_conv_0(tf.squeeze(tf.stack([self.bc_nn([left_boundary, self.dx]), self.flipud_method(self.rotation_method(self.bc_nn([right_boundary, self.dx]), k = 2)), self.fliplr_method(self.rotation_method(self.bc_nn([top_boundary, self.dx]), k = 1)), self.rotation_method(self.bc_nn([bottom_boundary, self.dx]), k = 3)], axis = self.stacking_dim), axis = 2))
        laplace_soln = self.post_bc_nn_conv_0(self.bc_nn([left_boundary, self.dx]) + self.flipud_method(self.rotation_method(self.bc_nn([right_boundary, self.dx]), k = 2)) + self.fliplr_method(self.rotation_method(self.bc_nn([top_boundary, self.dx]), k = 1)) + self.rotation_method(self.bc_nn([bottom_boundary, self.dx]), k = 3))
        homogeneous_poisson_soln = self.post_HPNN_conv_0(self.hpnn([rhs, self.dx]))
#         return homogeneous_poisson_soln + laplace_soln
        return self.post_merge_convolution_2(self.post_merge_convolution_1(self.post_merge_convolution_0(homogeneous_poisson_soln + laplace_soln)))
    
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
        quadrature_x, quadrature_w = tuple([np.polynomial.legendre.leggauss(self.n_quadpts)[i].astype(tf.keras.backend.floatx()) for i in range(2)])

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
        corner_coords = tf.cast(tf.gather_nd(tf.transpose(coords,(1,0,2)),corners), tf.keras.backend.floatx())
        
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