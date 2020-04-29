import tensorflow as tf
import itertools
import numpy as np
import multiprocessing
from collections.abc import Iterable

from ..utils import *

def handle_grid_parameters_range(grid_spacings_range, ndims):
    if not (isinstance(grid_spacings_range, tf.Tensor) or isinstance(grid_spacings_range, np.ndarray)):
        grid_spacings_range = tf.constant(grid_spacings_range,dtype=tf.keras.backend.floatx())
    if grid_spacings_range.shape == 2:
        assert grid_spacings_range[0] < grid_spacings_range[1], 'Upper bound for grid spacings must be larger than or equal to the lower bound!'
        grid_spacings_range = tf.expand_dims(grid_spacings_range, 0)
        grid_spacings_range = tf.tile(grid_spacings_range, [ndims,1])
    assert grid_spacings_range.shape[1] == 2, 'Dim 1 of grid_spacings range must be 2, containing the lower bound and upper bound for the respective spatial dimensions'
    assert grid_spacings_range.shape[0] == ndims, '1st dim of grid_spacings_range must have identical size to ndims'
    assert tf.reduce_all((grid_spacings_range[:,1] - grid_spacings_range[:,0]) >= 0.0), 'Dims ' + str(list(tf.reshape(tf.where((grid_spacings_range[:,1]-grid_spacings_range[:,0])<0), (-1,)).numpy())) + ' had upper bound of random range smaller than the lower bound'
    assert tf.reduce_all(grid_spacings_range[:,0] >= 0.0), 'Dims ' + str(list(tf.reshape(tf.where(grid_spacings_range[:,0]<0), (-1,)).numpy())) + ' had lower bounds of random range below 0'
    return grid_spacings_range #shape of grid_spacings_range is (ndims,2)


def build_fd_coefficients(stencil_size, orders, ndims):
    #handle orders input argument
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
    coefficients = np.zeros(stencil_size)
    slices = [list(stencil_size//2) for _ in range(ndims)]
    for dim in range(ndims):
        slices[dim][dim] = slice(0,stencil_size[dim])
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
    return tf.random.uniform(*args, dtype=tf.keras.backend.floatx())

@tf.function
def generate_random_grids(shapes):
    return tf.map_fn(generate_single_random_grid, shapes, back_prop = False, dtype = tf.keras.backend.floatx(), parallel_iterations = 32)

class reverse_poisson_dataset_generator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, batches_per_epoch, output_size_range, control_pt_grid_size_range, grid_spacings_range, ndims = 2, stencil_size = 5, solution_smoothness_range = None, homogeneous_bc = False, return_rhses = True, return_boundaries = True, return_dx = True, max_magnitude = 1.0):
        self.batch_size = batch_size
        self.ndims = ndims
        self.batches_per_epoch = batches_per_epoch
        self.grid_spacings_range = handle_grid_parameters_range(grid_spacings_range, ndims)

        self.output_size_range = handle_grid_parameters_range(output_size_range, ndims)
        self.control_pt_grid_size_range = handle_grid_parameters_range(control_pt_grid_size_range, ndims)

        #build fd stencil
        self.stencil = tf.constant(build_fd_coefficients(stencil_size, 2, ndims), dtype=tf.keras.backend.floatx())
        self._convolution_input_size_reduction_amount = (tf.constant(self.stencil.shape)//2)*2 #convolving an input with the stencil would reduce the input shape of spatial dimensions by the amounts stored in this variable
        
        #prepare padding variables if homogeneous bc is required
        self.homogeneous_bc = homogeneous_bc
        if self.homogeneous_bc:
            self._homogeneous_bc_control_pt_grid_paddings = tf.constant([[0,0],[0,0]] + [[1,1] for _ in range(ndims)], dtype=tf.int32)
            soln_grid_padding_for_one_side = tf.constant([0,0] + list((tf.constant(self.stencil.shape)//2).numpy()), dtype=tf.int32)
            self._homogeneous_bc_soln_grid_paddings = tf.stack([soln_grid_padding_for_one_side, soln_grid_padding_for_one_side],1)
            
        #set max magnitude for outputs
        self.max_magnitude = max_magnitude

        #set correct conv method to use when building rhs from solution
        self.conv_method = choose_conv_method(ndims)

        #adjust stencil shape for use with the conv method
        self.stencil = tf.expand_dims(tf.expand_dims(self.stencil, -1), -1)

        #build slice objects to recover solutions and BCs
        self._soln_slice = [Ellipsis] + [slice(int(self._convolution_input_size_reduction_amount[k]/2),-int(self._convolution_input_size_reduction_amount[k]/2)) for k in range(int(self._convolution_input_size_reduction_amount.shape[0]))]

        self.return_rhses = return_rhses
        if self.return_rhses:
            self._rhs_slice = [Ellipsis]

        self.return_boundaries = return_boundaries
        if self.return_boundaries:
            pass
                    
        

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
        #grid sizes shape: (batch_size, ndims)
        grid_sizes = tf.random.uniform((1,self.ndims), dtype = tf.keras.backend.floatx())
        grid_sizes = tf.tile(grid_sizes, [self.batch_size, 1])
        grid_sizes = tf.einsum('ij,j->ij', grid_sizes, grid_size_range[:,1] - grid_size_range[:,0]) + grid_size_range[:,0] + 1
        grid_sizes = tf.cast(grid_sizes, tf.int32)
        return grid_sizes

    @tf.function
    def adjust_grid_sizes_for_operator_application(self, grid_sizes):
        return grid_sizes + self._convolution_input_size_reduction_amount
    
    @tf.function
    def generate_soln(self):
        low_res_grid_sizes = self.generate_grid_sizes(self.control_pt_grid_size_range)
        control_pts = tf.expand_dims(generate_random_grids(low_res_grid_sizes),1)
        if self.homogeneous_bc:#if homogeneous bcs are desired, after generating the control pt grid, pad it with 0s, upsample, pad again with 0s to the same shape the soln would have taken otherwise
            control_pts = tf.pad(control_pts, self._homogeneous_bc_control_pt_grid_paddings, mode='CONSTANT', constant_values=0)
            soln_grid_size = self.generate_grid_sizes(self.output_size_range)
        else:
            soln_grid_size = self.adjust_grid_sizes_for_operator_application(self.generate_grid_sizes(self.output_size_range))
        soln_values = image_resize(control_pts, soln_grid_size,data_format = 'channels_first')
        if self.homogeneous_bc:
            soln_values = tf.pad(soln_values, self._homogeneous_bc_soln_grid_paddings, mode='CONSTANT', constant_values = 0)
        return soln_values

    @tf.function
    def generate_rhses_from_solutions(self, solutions):
        
        return self.conv_method(solutions, self.stencil, data_format = 'channels_first')

    @tf.function
    def __getitem__(self, idx = 0):
        solns = self.generate_soln()
        rhses = self.generate_rhses_from_solutions(solns)
        return solns[self._soln_slice],rhses

if __name__=='__main__':
    print(build_fd_coefficients([5,7,5],3,3))
    rpdg = reverse_poisson_dataset_generator(10,10,[[175,250],[100,150]],[[10,15],[10,15]],[1/249,1/99], homogeneous_bc = True, stencil_size = 5)
    print(rpdg.grid_spacings_range)
    print(rpdg.output_size_range)
    print(rpdg.generate_grid_spacings())
    gs = rpdg.generate_grid_sizes(rpdg.output_size_range)
    print(gs)
    ags = rpdg.adjust_grid_sizes_for_operator_application(gs)
    print(ags)
    print(generate_random_grids(ags).shape)
    print(rpdg.generate_soln().shape)
    print([x.shape for x in rpdg.__getitem__()])
        
