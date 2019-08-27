import tensorflow as tf
import itertools
import opt_einsum as oe
import numpy as np
from abc import ABC, abstractmethod

class Model_With_Integral_Loss_ABC(ABC, tf.keras.models.Model):
    '''
    Abstract class to implement integral loss.
    '''
    def __init__(self, mae_component_weight = 0.0, mse_component_weight = 0.0, n_quadpts= 20, Lp_norm_power = 2, **kwargs):
        super().__init__( **kwargs)
        self.mae_component_weight = mae_component_weight
        self.mse_component_weight = mse_component_weight
        self.n_quadpts = n_quadpts
        self.p = Lp_norm_power

    def integral_loss(self, y_true, y_pred):
        try: #try except block to handle keras model init shenanigans
            if self.data_format == 'channels_first':
                c = 0.5* tf.concat([self.dx * (int(y_true.shape[-2])-1), self.dx * (int(y_true.shape[-1])-1)],1)
                coords = np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-2]),np.linspace(-1, 1, y_true.shape[-1]),indexing = 'ij'), dtype = tf.keras.backend.floatx()).transpose((1,2,0)) #coordinates of each grid pt in the domain
            else:
                c = 0.5 * tf.concat([self.dx * int(y_true.shape[-3]), self.dx * int(y_true.shape[-2])],1)
                coords = np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-3]),np.linspace(-1, 1, y_true.shape[-2]),indexing = 'ij'), dtype = tf.keras.backend.floatx()).transpose((1,2,0)) #coordinates of each grid pt in the domain
        except:
            return 0.0*(y_true - y_pred)
        image_coords = [coords[:,0,0], coords[1,:,1]] #x and y coordinates separately
        quadrature_x, quadrature_w = tuple([x.astype(tf.keras.backend.floatx()) for x in np.polynomial.legendre.leggauss(self.n_quadpts)])
        #quadrature_x, quadrature_w = tuple([np.polynomial.legendre.leggauss(self.n_quadpts)[i].astype(np.float64) for i in range(2)])

        quadpts = tf.constant(np.array(np.meshgrid(quadrature_x,quadrature_x,indexing = 'ij')).transpose((1,2,0)),dtype = tf.keras.backend.floatx())
        #quadweights = tf.reduce_prod(c)*tf.tensordot(tf.squeeze(quadrature_w),tf.squeeze(quadrature_w),axes = 0)
        indices = [[],[]] #indices between each quadrature point lies - indices[0] is in x-dir and indices[1] is in the y-dir
        quad_coords = [quadpts[:,0,0], quadpts[1,:,1]] #x and y coordinates of each quad pt respectively
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
        corner_coords = tf.gather_nd(tf.transpose(coords,(0,1,2)),corners)
        
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

        inverse_domain_area = 1/tf.reduce_prod(2*c, axis = 1)

        #loss = tf.reduce_sum(tf.multiply(quadweights, values_at_quad_pts**self.p), axis = (0,1))**(1/self.p)
        loss = tf.reduce_mean(oe.contract('i,i...->i...',inverse_domain_area,tf.reduce_sum(tf.multiply(quadweights, values_at_quad_pts**self.p), axis = (0,1)), backend = 'tensorflow')**(1/self.p))
        
        if self.mae_component_weight != 0.0:
            loss = loss + self.mae_component_weight * tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))
        if self.mse_component_weight != 0.0:
            loss = loss + self.mse_component_weight * tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        return loss
