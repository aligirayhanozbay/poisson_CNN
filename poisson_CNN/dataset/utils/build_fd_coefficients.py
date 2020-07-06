import numpy as np
import tensorflow as tf
from .get_fd_coefficients import get_fd_coefficients

def build_fd_coefficients(stencil_size, orders, ndims = None):
    '''
    Builds ndims ndims-dimensional tensors, where the kth tensor contains the finite difference coefficients with stencil_size[k] points for derivatives of orders[k] along the kth direction.

    Inputs:
    -stencil_size: int or list of ints, np.array or tf.Tensor. Number of stencil points per direction.
    -orders: int or list of ints. Order of the derivative to be computed per direction.
    -ndims: int. Number of spatial dimensions. Has to be supplied if stencil_size is an int.
    '''
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
