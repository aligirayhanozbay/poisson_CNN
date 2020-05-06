import tensorflow as tf
import string
import math

@tf.function
def generate_smooth_function(grid_size,coefficients_size,homogeneous_bc = False):
    '''
    Generates a smooth function sampled on a grid the size of which is given by grid_size.

    Inputs:
    -grid_size: list of ints. determines the shape of the output
    -coefficients_size: list of ints. determines the number of Fourier coefficients to use per spatial dim. larger values result in less smooth functions
    -homogeneous_bc: bool. If set to true, the value of the result on the boundaries will be 0.0

    Outputs:
    -result: the generated smooth function.
    '''
    ndims = len(grid_size)
    coefficients = 2*tf.random.uniform(coefficients_size)-1
    coords = [tf.linspace(0.0,math.pi,grid_size[k]) for k in range(ndims)]
    wavenumbers = [tf.cast(tf.range(1,coefficients_size[k]+1),tf.keras.backend.floatx()) for k in range(ndims)]
    
    trig_arguments = [tf.einsum('i,j->ij',coord,wavenumber) for coord,wavenumber in zip(coords,wavenumbers)]
    trig_values = [tf.sin(x)+tf.cos(x) if not homogeneous_bc else tf.sin(x) for x in trig_arguments] 
    
    einsum_str = [string.ascii_uppercase[:ndims]] + [string.ascii_lowercase[k] + string.ascii_uppercase[k] for k in range(ndims)]
    einsum_str = ','.join(einsum_str) + '->' + string.ascii_lowercase[:ndims]
    res = tf.einsum(einsum_str,coefficients,*trig_values)
    return res/tf.reduce_max(tf.abs(res))
