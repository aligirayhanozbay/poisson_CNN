import tensorflow as tf
import numpy as np

@tf.function
def find_neighbouring_indices_along_axis(grid_sizes, quadrature_coords_along_axis, dim):
    dim_size = grid_sizes[dim]
    dx = tf.cast(2/(dim_size-1), tf.keras.backend.floatx())
    lower_idx = tf.cast(tf.math.floor((quadrature_coords_along_axis+1)/dx), tf.int32)
    upper_idx = tf.cast(tf.math.ceil((quadrature_coords_along_axis+1)/dx), tf.int32)
    equal_indices = (upper_idx == lower_idx)
    if tf.reduce_any(equal_indices):#sometimes a quad pt may coincide with the position of a gridpoint. adjust upper idx for that possibility
        new_upper_indices = tf.gather_nd(upper_idx,tf.where(equal_indices)) + 1
        upper_idx = tf.tensor_scatter_nd_update(upper_idx, tf.where(equal_indices), new_upper_indices)
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
        linspace_start  = tf.constant(-1.0,tf.keras.backend.floatx())
        linspace_end = tf.constant(1.0,tf.keras.backend.floatx())
        for k in range(loop_start, loop_start+ndims):
            return_list.append(tf.linspace(linspace_start,linspace_end,tf.cast(gridshape[k], tf.int32)))
        return return_list
    return get_input_grid_coordinates

@tf.function
def expand_1d_tensor_to_k_dims(tensor, dim, ndims):
    newshape = [1 for _ in range(dim)] + [-1] + [1 for _ in range(ndims-dim-1)]
    return tf.reshape(tensor,newshape)

@tf.function
def tile_1d_tensor_to_shape(tensor, dim, newshape):
    tilings = tf.concat([newshape[:dim], tf.constant([1], dtype = tf.int32), newshape[dim+1:]],0)
    return tf.tile(tensor,tilings)

@tf.function
def build_corner_coordinate_combinations(corner_coords, ndims):
    resulting_shape = [tf.shape(x)[0] for x in corner_coords]
    corner_coords = [expand_1d_tensor_to_k_dims(coords,k,ndims) for k,coords in enumerate(corner_coords)]
    corner_coords = [tile_1d_tensor_to_shape(coords,k,resulting_shape) for k,coords in enumerate(corner_coords)]
    return tf.stack(corner_coords,-1)

def binary_numbers_up_to_value(k):
    log2 = tf.cast(tf.math.log(2.0), tf.keras.backend.floatx())
    bits = int(tf.cast(tf.math.ceil(tf.math.log(tf.cast(k,tf.keras.backend.floatx()))/log2),tf.int32))
    return np.array([list(reversed([int(s) for s in list(('{0:0' + str(bits) + 'b}').format(i))])) for i in range(k)])

@tf.function
def sample_values_enclosing_GL_quadrature_points_from_grid(*neighbouring_indices, grid, corner_labels, data_format):
    target_shape = [indices.shape[0] for indices in neighbouring_indices]
    tiled_indices = [tf.reshape(neighbouring_indices[k],[1 for _ in range(k)]+[neighbouring_indices[k].shape[0]]+[1 for _ in range(len(grid.shape)-k-3)]+[2]) for k in range(len(neighbouring_indices))]
    tiled_indices = [tf.tile(indices,target_shape[:k] + [1] + target_shape[k+1:] + [1]) for k,indices in enumerate(tiled_indices)]
    indices = tf.transpose(tf.stack(tiled_indices,axis=-2),[len(grid.shape)-2,len(grid.shape)-1]+[k for k in range(len(grid.shape)-2)])
    indices = tf.map_fn(lambda x: tf.gather_nd(indices,x),corner_labels,dtype=indices.dtype)
    indices = tf.transpose(indices,[2+k for k in range(len(grid.shape)-2)] + [0,1])
    if data_format == 'channels_last':
        grid = tf.transpose(grid,[0,len(grid.shape)-1] + [k for k in range(1,len(grid.shape)-2)])
    values=tf.map_fn(lambda x: tf.map_fn(lambda y: tf.gather_nd(y,indices),x,dtype=x.dtype),grid,dtype=grid.dtype)
    if data_format == 'channels_last':
        values = tf.transpose(values,[0] + [k for k in range(2,len(grid.shape)-1)] + [1])

    return values
    

class integral_loss(tf.keras.losses.Loss):
    def __init__(self, n_quadpts, ndims = None, Lp_norm_power = 2, data_format = 'channels_first', reduce_results = False):
        '''
        This is a loss function which calculates the integral (or continuous) Lp norm of the ground truth and the prediction instead of the discrete Lp norm. (y-t)^p is interpolated onto the Gauss-Legendre quadrature points using multilinear interpolation and then the integral of the quantity over the domain is approximated with the GL quadrature method.

        Init arguments:
        -n_quadpts: int or tuple of ints. Determines the number of quadrature points along each dimension.
        -ndims: int. Number of spatial dimensions. Necessary only if an int value is passed as n_quadpts.
        -Lp_norm_power: int. Determines the order of the norm.
        -data_format: str. see tf.keras documentation.
        -reduce_results: bool. If true, compute mean of per-sample and per-channel losses and return a scalar.
        '''
        super().__init__()
        if ndims is None:
            ndims = len(n_quadpts)
        if isinstance(n_quadpts, int):
            n_quadpts = [n_quadpts for _ in range(ndims)]
        self.n_quadpts = n_quadpts
        self.ndims = ndims
        self.Lp_norm_power = Lp_norm_power
        self.data_format = data_format
        self.reduce_results = reduce_results

        quadrature_coords = []
        quadrature_weights = []
        for dim in range(self.ndims):
            x, w = tuple([x.astype(tf.keras.backend.floatx()) for x in np.polynomial.legendre.leggauss(self.n_quadpts[dim])])
            quadrature_coords.append(tf.cast(x,tf.keras.backend.floatx()))
            quadrature_weights.append(tf.cast(w,tf.keras.backend.floatx()))

        self.quadrature_coords_meshgrid = tf.stack(tf.meshgrid(*quadrature_coords,indexing='ij'),0)
        self.quadrature_weights = tf.reduce_prod(tf.stack(tf.meshgrid(*quadrature_weights,indexing='ij'),0),0)

        self.find_neighbouring_indices = find_neighbouring_indices_wrapper(quadrature_coords, self.data_format)
        self.get_input_grid_coordinates = get_input_grid_coordinates_wrapper(self.ndims, self.data_format)
        self.corner_indices = tf.constant(binary_numbers_up_to_value(2**self.ndims),dtype=tf.int64)
        
        self.multilinear_interpolation_basis_polynomial_values_at_quadrature_coords = tf.map_fn(lambda x: tf.reduce_prod(tf.boolean_mask(self.quadrature_coords_meshgrid, x),0),self.corner_indices,dtype=self.quadrature_coords_meshgrid.dtype)

        self.losses_einsum_str = 'ij...,...->ij...' if self.data_format == 'channels_first' else 'i...j,...->ij...'

    @tf.function
    def get_interpolation_matrix_row(self,corner_coord):
        return tf.map_fn(lambda x: tf.reduce_prod(tf.boolean_mask(corner_coord,x),axis=0),self.corner_indices,dtype=corner_coord.dtype)
        
    @tf.function
    def __call__(self, y_true, y_pred):
        '''
        Calculate the Lp norm.

        Inputs:
        -y_true: tf.Tensor of shape (batch_size,n_channels,...) or (batch_size,...,n_channels) depending on self.data_format. Ground truth.
        -y_pred: tf.Tensor of shape identical to y_true or a list of 2 tensors, first of which has shape identical to y_true and second is of shape (batch_size,self.ndims). Contains the prediction and, optionally, a grid spacing value that can be used to adjust loss calculation to domains other than the default [-1,1] x ... x [-1,1]
        '''
        gridshape = tf.shape(y_true)
        if isinstance(y_pred,list) and len(y_pred) == 2:
            y_pred, dx = y_pred
            domain_size = tf.einsum('i,bi->bi',tf.cast((gridshape[2:] if self.data_format == 'channels_first' else gridshape[1:-1])-1,dx.dtype),dx)
        else:
            domain_size = 2*tf.ones([gridshape[0],self.ndims])
        
        neighbouring_indices = self.find_neighbouring_indices(gridshape)
        input_grid_coordinates = self.get_input_grid_coordinates(gridshape)

        #Build linear interpolation LHS matrix
        lower_corner_coords = [tf.gather_nd(input_grid_coordinates[k], tf.expand_dims(neighbouring_indices[k][:,0],1)) for k in range(self.ndims)]
        lower_corner_coords = build_corner_coordinate_combinations(lower_corner_coords,self.ndims)#coordinates (x0,y0...) of the rectangle bounding each quadrature point
        upper_corner_coords = [tf.gather_nd(input_grid_coordinates[k], tf.expand_dims(neighbouring_indices[k][:,1],1)) for k in range(self.ndims)]
        upper_corner_coords = build_corner_coordinate_combinations(upper_corner_coords,self.ndims)#coordinates (x1,y1,...) of the rectangle bounding each quadrature point

        corner_coords = tf.stack([lower_corner_coords, upper_corner_coords],-1)
        corner_labels = tf.stack([tf.tile(tf.constant([[k for k in range(self.ndims)]]),tf.constant([2**self.ndims,1])),tf.cast(self.corner_indices,tf.int32)],-1)#contains labels for each corner of the rectangle such as [0,0,0],[0,1,0],[1,1,0] etc
        transpose_axes = [self.ndims,self.ndims+1] + [k for k in range(self.ndims)]
        corner_coords_transposed = tf.transpose(corner_coords,transpose_axes)
        corner_coords = tf.map_fn(lambda x: tf.gather_nd(corner_coords_transposed,x),corner_labels,dtype=corner_coords_transposed.dtype)#get coordinates belonging to each corner of the n dimensional rectangle enclosing each quadrature point
        interpolation_matrix = tf.map_fn(self.get_interpolation_matrix_row,corner_coords,dtype=corner_coords.dtype)
        interpolation_matrix = tf.transpose(interpolation_matrix,[2+k for k in range(self.ndims)] + [0,1])#interpolation LHS matrix
        interpolation_matrix = tf.expand_dims(tf.expand_dims(interpolation_matrix,0),0)
        #interpolation_matrix = tf.tile(interpolation_matrix,[y_true.shape[0]] + [1 for _ in interpolation_matrix.shape[1:]])
        interpolation_matrix = tf.keras.backend.tile(interpolation_matrix,[tf.keras.backend.shape(y_true)[0]] + [1 for _ in interpolation_matrix.shape[1:]])
        
        #build RHS
        pointwise_loss = (y_true - y_pred)**self.Lp_norm_power
        pointwise_loss_values_surrounding_quadrature_points = tf.expand_dims(sample_values_enclosing_GL_quadrature_points_from_grid(*neighbouring_indices, grid=pointwise_loss, corner_labels=corner_labels, data_format=self.data_format),-1)

        #solve linear systems to acquire interpolation coefficients
        interpolation_coefficients = tf.linalg.solve(interpolation_matrix,pointwise_loss_values_surrounding_quadrature_points)[...,0]
        #get loss at quadrature points
        loss_at_quadrature_points_einsum_str = 'i...,bc...i->bc...' if self.data_format == 'channels_first' else 'i...,b...ci->b...c'
        loss_at_quadrature_points = tf.einsum(loss_at_quadrature_points_einsum_str,self.multilinear_interpolation_basis_polynomial_values_at_quadrature_coords, interpolation_coefficients)

        #integrate using GL quadrature
        losses = tf.einsum(self.losses_einsum_str,loss_at_quadrature_points, self.quadrature_weights)
        losses = tf.reduce_sum(losses, axis = [k for k in range(2,self.ndims+2)])
        losses = tf.einsum('ij,i->ij',losses,tf.reduce_prod(domain_size,axis=1))/(2**self.ndims)

        if self.reduce_results:
            losses = tf.reduce_mean(losses)
        
        return losses
    
if __name__=='__main__':
    print('---Integral loss unit test---')
    print('Integrate (xyz)^(2/3) in the domain [[0,1],[0,2],[1,3.5]]')
    loss_func = integral_loss((25,13,28), ndims = 3)
    print('Integral loss instantiated')
    x = tf.linspace(0.0,1.0,150)
    y = tf.linspace(0.0,2.0,200)
    z = tf.linspace(1.0,3.5,175)

    t = tf.einsum('i,j,k->ijk',x,y,z)**(1/3)
    t = tf.expand_dims(tf.expand_dims(t,0),0)
    dx = tf.stack([[x[1]-x[0],y[1]-y[0],z[1]-z[0]]])

    u = tf.zeros(t.shape)
    print('Test variables created')
    
    dc = loss_func(t,[u,dx])
    print('Integration calculation ran.')
    true_soln = 4.84711
    assert dc.shape == tf.TensorShape([t.shape[0],t.shape[1]])
    print('Output shape correct: ' + str(dc.shape))
    assert tf.abs((true_soln - dc[0,0])/true_soln) < 0.01
    print('Value correctly computed. Calculated value: ' + str(dc.numpy()[0,0]) + ' | Target value: ' + str(true_soln) + ' | Error: ' + str(tf.abs((true_soln - dc[0,0])/true_soln).numpy()) + '%')
