import tensorflow as tf
import tensorflow_probability as tfp
import itertools
import numpy as np
import multiprocessing
from collections.abc import Iterable

from ..utils import *

def handle_grid_parameters_range(value_range, ndims):
    if not (isinstance(value_range, tf.Tensor) or isinstance(value_range, np.ndarray)):
        value_range = tf.constant(value_range,dtype=tf.keras.backend.floatx())
    if value_range.shape == 2:
        assert value_range[0] < value_range[1], 'Upper bound for grid spacings must be larger than or equal to the lower bound!'
        value_range = tf.expand_dims(value_range, 0)
        value_range = tf.tile(value_range, [ndims,1])
    assert value_range.shape[1] == 2, 'Dim 1 of grid_spacings range must be 2, containing the lower bound and upper bound for the respective spatial dimensions'
    assert value_range.shape[0] == ndims, '1st dim of value_range must have identical size to ndims'
    assert tf.reduce_all((value_range[:,1] - value_range[:,0]) >= 0.0), 'Dims ' + str(list(tf.reshape(tf.where((value_range[:,1]-value_range[:,0])<0), (-1,)).numpy())) + ' had upper bound of random range smaller than the lower bound'
    assert tf.reduce_all(value_range[:,0] >= 0.0), 'Dims ' + str(list(tf.reshape(tf.where(value_range[:,0]<0), (-1,)).numpy())) + ' had lower bounds of random range below 0'
    return value_range #shape of value_range is (ndims,2)


def build_fd_coefficients(stencil_size, orders, ndims = None):
    #handle orders input argument
    if ndims is None:
        ndims = len(stencil_size)
    if isinstance(orders, int):
        orders = [int(orders) for _ in range(ndims)]
    else:
        orders = [int(order) for order in orders]
    assert np.all(np.array(orders) > 0), 'All derivative orders must be positive'
    #convert stencil size to numpy
    if isinstance(stencil_size, int):
        stencil_size = [stencil_size]
    if isinstance(stencil_size, list) or isinstance(stencil_size, tuple):
        stencil_size = np.array(stencil_size)
    elif isinstance(stencil_size, tf.Tensor):
        stencil_size = stencil_size.numpy()
    #assert a stencil size exists for each dim
    if stencil_size.shape[0] == 1:
        stencil_size = np.repeat(stencil_size, ndims)
    assert len(stencil_size) == ndims
    assert np.all((stencil_size%2) == 1), 'Stencil sizes must be all odd - this program uses symmetric stencils. Stencil sizes supplied were: ' + str(stencil_size)

    #build coefficients
    coefficients = np.zeros(np.insert(stencil_size,0,ndims))
    slices = [[dim] + list(stencil_size//2) for dim in range(ndims)]
    for dim in range(ndims):
        slices[dim][dim+1] = slice(0,stencil_size[dim])
        stencil_positions = list(np.arange(-stencil_size[dim]//2+1,stencil_size[dim]//2+1))
        coefficients[tuple(slices[dim])] += get_fd_coefficients(stencil_positions, orders[dim])
    return coefficients

def choose_conv_method(ndims):
    if ndims == 1:
        return tf.keras.backend.conv1d
    elif ndims == 2:
        return tf.keras.backend.conv2d
    elif ndims == 3:
        return tf.keras.backend.conv3d
    else:
        raise(NotImplementedError('Convolutions above 3D are not available yet'))

@tf.function
def generate_single_random_grid(*args):
    return 2*tf.random.uniform(*args, dtype=tf.keras.backend.floatx())-1

@tf.function
def generate_random_grids(grid_size,k):
    grid_sizes = tf.tile(tf.expand_dims(grid_size,0),[k,1])
    return tf.map_fn(generate_single_random_grid, grid_sizes, back_prop = False, dtype = tf.keras.backend.floatx(), parallel_iterations = 32)

class reverse_poisson_dataset_generator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, batches_per_epoch, output_size_range, coeff_grid_size_range, grid_spacings_range = None, normalize_domain_size = False, ndims = None, stencil_size = 5, homogeneous_bc = False, return_rhses = True, return_boundaries = True, return_dx = True, max_magnitude = 1.0):
        '''
        Generates batches of random Poisson equation RHS-BC-solutions by first generating a solution and then using finite difference schemes to generate the RHS. Smooth results are ensured by generating random solutions on a low res control pt grid and then using cubic/bi-cubic/tri-cubic upsampling to the target resolution

        Inputs:
        -batch_size: int. Number of samples to generate each time __getitem__ is called
        -batches_per_epoch: int. Determines the number of batches to generate per keras epoch. New, random data are generated each time __getitem__ is called, so this is mostly for compatibility purposes
        -output_size_range: List of 2 ints, list of list of 2 ints, or np.ndarray/tf.Tensor of shape (ndims,2). Determines the range of random values from which the size of each spatial dimension will be picked for each batch (i.e. random value range for the shape of the  spatial dimensions)
        -control_pt_grid_size_range: List of 2 ints, list of list of 2 ints, or np.ndarray/tf.Tensor of shape (ndims,2). Same as output_size range, but determines the range for the control pt grid size. Higher values create less smooth outputs.
        -grid_spacings_range:  List of 2 floats, list of list of 2 floats, or np.ndarray/tf.Tensor of shape (ndims,2). Determines the range of values that the grid spacings can take for each dimension.
        -ndims: int. Number of spatial dimensions.
        -stencil_size: Odd int or list of odd ints. Determines the size of the FD stencil to be used for each dimension.
        -homogeneous_bc: bool. If set to true, solutions with homogeneous BCs will be returned only.
        -return_rhses: bool. If set to true, RHSes will be returned
        -return_boundaries: bool. If set to true, BCs will be returned.
        -return_dx: bool. If set to true, grid spacings will be returned.
        -max_magnitude:
        '''
        self.batch_size = batch_size
        if ndims is None:
            for k in [output_size_range,coeff_grid_size_range]:
                try:
                    ndims = len(k)
                except:
                    pass
        self.ndims = ndims
        self.batches_per_epoch = batches_per_epoch

        self.grid_spacings_range = handle_grid_parameters_range(grid_spacings_range, ndims)
        self.output_size_range = handle_grid_parameters_range(output_size_range, ndims)
        self.coeff_grid_size_range = handle_grid_parameters_range(coeff_grid_size_range, ndims)

        #build fd stencil
        self.stencil = tf.constant(build_fd_coefficients(stencil_size, 2, ndims), dtype=tf.keras.backend.floatx())
        self._convolution_input_size_reduction_amount = (tf.constant(self.stencil.shape[1:])//2)*2 #convolving an input with the stencil would reduce the input shape of spatial dimensions by the amounts stored in this variable
        
        #prepare padding variables if homogeneous bc is required
        self.homogeneous_bc = homogeneous_bc
            
        #set max magnitude for outputs
        self.max_magnitude = max_magnitude

        #set correct conv method to use when building rhs from solution
        self.conv_method = choose_conv_method(ndims)

        #build slice objects to recover solutions and BCs
        self._soln_slice = [Ellipsis] + [slice(int(self._convolution_input_size_reduction_amount[k]/2),-int(self._convolution_input_size_reduction_amount[k]/2)) for k in range(int(self._convolution_input_size_reduction_amount.shape[0]))]

        self.return_rhses = return_rhses

        self.return_boundaries = return_boundaries
        if self.return_boundaries:
            self._boundary_slices = []
            for k in range(self.ndims):
                sl0 = [Ellipsis] + [slice(0,None) for k in range(k)] + [0] + [slice(0,None) for k in range(ndims-k-1)]
                sl1 = [Ellipsis] + [slice(0,None) for k in range(k)] + [-1] + [slice(0,None) for k in range(ndims-k-1)]
                self._boundary_slices.append(sl0)
                self._boundary_slices.append(sl1)

        self.return_dx = return_dx
        self.normalize_domain_size = normalize_domain_size
                    
        

    def __len__(self):
        return self.batches_per_epoch

    @tf.function
    def generate_grid_spacings(self):
        #grid spacings shape: (batch_size, ndims)
        grid_spacings = tf.random.uniform((self.batch_size, self.ndims), dtype = tf.keras.backend.floatx())
        grid_spacings = tf.einsum('ij,j->ij', grid_spacings, self.grid_spacings_range[:,1] - self.grid_spacings_range[:,0]) + self.grid_spacings_range[:,0]
        return grid_spacings

    @tf.function
    def generate_grid_sizes(self, grid_size_range):
        #grid sizes shape: (ndims,)
        grid_sizes = tf.random.uniform((self.ndims,), dtype = tf.keras.backend.floatx())
        grid_sizes = (grid_size_range[:,1] - grid_size_range[:,0]) * grid_sizes + grid_size_range[:,0] + 1
        grid_sizes = tf.cast(grid_sizes, tf.int32)
        return grid_sizes

    @tf.function
    def adjust_grid_sizes_for_operator_application(self, grid_sizes):
        return grid_sizes + self._convolution_input_size_reduction_amount
    
    @tf.function
    def generate_soln(self):
        tiles = [self.batch_size] + [1 for _ in range(self.ndims)]

        coeff_grid_size_range = tf.expand_dims(self.coeff_grid_size_range,0)
        coeff_grid_size_range = tf.tile(coeff_grid_size_range,tiles)
        n_coefficients = tf.map_fn(self.generate_grid_sizes,coeff_grid_size_range,dtype=tf.int32)

        output_shape = self.generate_grid_sizes(self.output_size_range)
        output_shape = tf.expand_dims(output_shape,0)
        output_shape = tf.tile(output_shape,tiles[:2])

        solns = tf.map_fn(lambda x: generate_smooth_function(x[0],x[1],self.homogeneous_bc),(output_shape,n_coefficients),dtype=tf.keras.backend.floatx())
        
        return tf.expand_dims(solns,1)
        

    @tf.function
    def generate_rhses_from_solutions(self, solutions):
        grid_spacings = self.generate_grid_spacings()
        if self.normalize_domain_size:
            domain_sizes = tf.einsum('ij,j->ij',grid_spacings,tf.cast(tf.shape(solutions[self._soln_slice])[2:],grid_spacings.dtype))
            max_domain_size = tf.reduce_max(domain_sizes,1)
            grid_spacings = tf.einsum('ij,i->ij',grid_spacings,1/max_domain_size)
        finite_difference_conv_kernels = tf.einsum('ij,j...->i...',1/(grid_spacings**2),self.stencil)#convert stencil to kernels
        finite_difference_conv_kernels = tf.expand_dims(tf.expand_dims(finite_difference_conv_kernels, -1), -1)#adjust kernel dims for use with the conv method
        rhses = tf.map_fn(lambda x: self.conv_method(x=x[0], kernel=x[1], data_format = 'channels_first')[0],(tf.expand_dims(solutions,1), finite_difference_conv_kernels), dtype=tf.keras.backend.floatx())#perform the conv for each
        return rhses, grid_spacings

    @tf.function
    def pack_outputs(self, rhses, solns, grid_spacings):
        solns = solns[self._soln_slice]

        problem_definition = []
        if self.return_rhses:
            problem_definition.append(rhses)

        if self.return_boundaries:
            for sl in self._boundary_slices:
                problem_definition.append(solns[sl])

        if self.return_dx:
            problem_definition.append(grid_spacings)

        return problem_definition, solns

    @tf.function
    def __getitem__(self, idx = 0):
        solns = self.generate_soln()
        rhses, grid_spacings = self.generate_rhses_from_solutions(solns)
        out = self.pack_outputs(rhses, solns, grid_spacings)
        return out

if __name__=='__main__':
    nmax = 1500
    nmin = 50
    dmax = 1*1/(nmin-1)
    dmin = 0.5*1/(nmax-1)
    dxrange = [dmin,dmax]
    ndims = 2
    grid_size_range = [[nmin,nmax] for _ in range(ndims)]
    cmax = 10
    cmin = 5
    ctrl_pt_range = [[cmin,cmax] for _ in range(ndims)]
    rpdg = reverse_poisson_dataset_generator(10,10,grid_size_range,ctrl_pt_range,dxrange,homogeneous_bc = True, stencil_size = 5,normalize_domain_size = True)
    for k in range(10):
        inp, out = rpdg.__getitem__()
        print('---')
        print('max soln val ' + str(tf.reduce_max(tf.abs(out))))
        print('max soln val divided by no of pts ' + str(tf.reduce_max(tf.abs(out))/tf.reduce_prod(tf.cast(tf.shape(inp[0])[2:],tf.float32))))
        print('max rhs val ' + str(tf.reduce_max(tf.abs(inp[0]))))
        print('max rhs val divided by no of pts ' + str(tf.reduce_max(tf.abs(inp[0]))/tf.reduce_prod(tf.cast(tf.shape(inp[0])[2:],tf.float32))))
        sisd = tf.map_fn(lambda x: tf.reduce_max(tf.abs(x)),inp[0]) / tf.reduce_sum(1/(inp[-1]**2),1)
        print('max rhs val divided by sum of inverse squares of dxes ' + str(sisd))
        print('shape ' + str(tf.shape(inp[0])[2:]))
    import pdb
    pdb.set_trace()
    '''
    rpdg = reverse_poisson_dataset_generator(10,10,[[175,250],[100,150]],[[10,15],[10,15]],[1/249,1/99], homogeneous_bc = True, stencil_size = 5)
    inp, out = rpdg.__getitem__()
    '''
    
