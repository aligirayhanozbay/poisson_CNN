import tensorflow as tf
import numpy as np

#@tf.function
def interpolate_grid_values_onto_points(grid_values, grid_coords, target_coords):
    '''
    Interpolates the values stored in grid_values at coordinates at grid_coords to the coordinates in target_coords.
    '''
    ndims = tf.shape(grid_coords)[0]

@tf.function
def find_neighbouring_indices_along_axis(grid_sizes, quadrature_coords_along_axis, dim):
    dim_size = grid_sizes[dim]
    dx = tf.cast(2/(dim_size-1), tf.keras.backend.floatx())
    lower_idx = tf.cast(tf.math.floor((quadrature_coords_along_axis+1)/dx), tf.int32)
    upper_idx = tf.cast(tf.math.ceil((quadrature_coords_along_axis+1)/dx), tf.int32)
    return tf.stack([lower_idx,upper_idx],1)
    
def find_neighbouring_indices_wrapper(quadrature_coords, data_format):
    @tf.function
    def find_neighbouring_indices(grid_shape):
        if data_format == 'channels_first':
            grid_shape = grid_shape[2:]
        elif data_format == 'channels_last':
            grid_shape = grid_shape[1:-1]
        return_list = []
        for k,item in enumerate(quadrature_coords):
            return_list.append(find_neighbouring_indices_along_axis(grid_shape, item, k))
        return return_list
    return find_neighbouring_indices

def get_input_grid_coordinates_wrapper(ndims, data_format):
    @tf.function
    def get_input_grid_coordinates(gridshape):
        loop_start = 2 if data_format == 'channels_first' else 1
        return_list = []
        for k in range(loop_start, loop_start+ndims):
            return_list.append(tf.linspace(-1.0,1.0,tf.cast(gridshape[k], tf.int32)))
        return return_list
    return get_input_grid_coordinates

@tf.function
def expand_1d_tensor_to_k_dims(tensor, dim, ndims):
    newshape = [1 for _ in range(dim)] + [-1] + [1 for _ in range(ndims-dim-1)]
    return tf.reshape(tensor,newshape)

@tf.function
def tile_1d_tensor_to_shape(tensor, dim, newshape):
    tilings = tf.concat([newshape[:dim], tf.constant([1], dtype = tf.int32), newshape[dim+1:]],0)#[1 for _ in range(dim)] + [-1] + [1 for _ in range(ndims-dim-1)]
    return tf.tile(tensor,tilings)

@tf.function
def build_corner_coordinate_combinations(corner_coords, ndims):
    resulting_shape = [tf.shape(x)[0] for x in corner_coords]
    corner_coords = [expand_1d_tensor_to_k_dims(coords,k,ndims) for k,coords in enumerate(corner_coords)]
    corner_coords = [tile_1d_tensor_to_shape(coords,k,resulting_shape) for k,coords in enumerate(corner_coords)]
    return tf.stack(corner_coords,-1)

'''
#@tf.function
def binary_numbers_up_to_value(k):
    bits = int(tf.cast(tf.math.ceil(tf.math.log(tf.cast(k,tf.keras.backend.floatx()))/tf.math.log(2.0)),tf.int32))
    return [[int(s) for s in list(('{0:0' + str(bits) + 'b}').format(i))] for i in range(k)]
'''
def binary_numbers_up_to_value(k):
    bits = int(tf.cast(tf.math.ceil(tf.math.log(tf.cast(k,tf.keras.backend.floatx()))/tf.math.log(2.0)),tf.int32))
    return np.array([list(reversed([int(s) for s in list(('{0:0' + str(bits) + 'b}').format(i))])) for i in range(k)])

class integral_loss:
    def __init__(self, n_quadpts, ndims = None, mae_component_weight = 0.0, mse_component_weight = 0.0, Lp_norm_power = 2, data_format = 'channels_first'):
        if ndims is None:
            ndims = len(n_quadpts)
        if isinstance(n_quadpts, int):
            n_quadpts = (n_quadpts for _ in range(ndims))
        self.n_quadpts = n_quadpts
        self.ndims = ndims
        self.mae_component_weight = mae_component_weight
        self.mse_component_weight = mse_component_weight
        self.p = Lp_norm_power
        self.data_format = data_format

        quadrature_coords = []
        quadrature_weights = []
        for dim in range(self.ndims):
            x, w = tuple([x.astype(tf.keras.backend.floatx()) for x in np.polynomial.legendre.leggauss(self.n_quadpts[dim])])
            quadrature_coords.append(tf.cast(x,tf.keras.backend.floatx()))
            quadrature_weights.append(tf.cast(w,tf.keras.backend.floatx()))

        #quadrature_coords_meshgrid = tf.stack(tf.meshgrid(*quadrature_coords),0)
        quadrature_weights = tf.reduce_prod(tf.stack(tf.meshgrid(*quadrature_weights),0),0)


        self.find_neighbouring_indices = find_neighbouring_indices_wrapper(quadrature_coords, self.data_format)
        self.get_input_grid_coordinates = get_input_grid_coordinates_wrapper(self.ndims, self.data_format)
        
    @tf.function
    def __call__(self,y_true, y_pred):
        #y_pred, dx = y_pred #unpack predictions
        gridshape = tf.shape(y_true)
        neighbouring_indices = self.find_neighbouring_indices(gridshape)
        input_grid_coordinates = self.get_input_grid_coordinates(gridshape)
        
        lower_corner_coords = [tf.gather_nd(input_grid_coordinates[k], tf.expand_dims(neighbouring_indices[k][:,0],1)) for k in range(self.ndims)]
        lower_corner_coords = build_corner_coordinate_combinations(lower_corner_coords,self.ndims)
        upper_corner_coords = [tf.gather_nd(input_grid_coordinates[k], tf.expand_dims(neighbouring_indices[k][:,1],1)) for k in range(self.ndims)]
        upper_corner_coords = build_corner_coordinate_combinations(upper_corner_coords,self.ndims)

        corner_coords = tf.stack([lower_corner_coords, upper_corner_coords],-1)
        corner_indices = tf.numpy_function(binary_numbers_up_to_value,[2**self.ndims],tf.int64)
        p = tf.stack([tf.tile(tf.constant([[k for k in range(self.ndims)]]),tf.constant([2**self.ndims,1])),tf.cast(corner_indices,tf.int32)],-1)
        transpose_axes = [self.ndims,self.ndims+1] + [k for k in range(self.ndims)]
        corner_coords_transposed = tf.transpose(corner_coords,transpose_axes)
        corner_coords = tf.map_fn(lambda x: tf.gather_nd(corner_coords_transposed,x),p,dtype=corner_coords_transposed.dtype)

        #@tf.function
        #def get_interpolation_matrix_rows()
        
        #interpolation_matrix = tf.einsum('ij...,kj->ik...',corner_coords,tf.cast(corner_indices,corner_coords.dtype))
        return corner_coords#interpolation_matrix
        '''
        if self.data_format == 'channels_first':
            c = 0.5 * tf.einsum('ij,j->ij',dx,tf.cast(gridshape[2:]-1,tf.keras.backend.floatx()))
            
            coords =
            #np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-2]),np.linspace(-1, 1, y_true.shape[-1]),indexing = 'ij'), dtype = tf.keras.backend.floatx()).transpose((1,2,0)) #coordinates of each grid pt in the domain
        else:
            c = 0.5 * tf.einsum('ij,j->ij',dx,tf.cast(gridshape[1:-1]-1,tf.keras.backend.floatx()))
            coords = #np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-3]),np.linspace(-1, 1, y_true.shape[-2]),indexing = 'ij'), dtype = tf.keras.backend.floatx()).transpose((1,2,0))
        '''

if __name__=='__main__':
    loss_func = integral_loss((10,15,25))
    t = tf.zeros((10,1,100,150,140), dtype = tf.keras.backend.floatx())
    y = t+1
    dc = loss_func(y,t)
    print(dc.shape)
    import pdb
    pdb.set_trace()
    print(binary_numbers_up_to_value(8))
