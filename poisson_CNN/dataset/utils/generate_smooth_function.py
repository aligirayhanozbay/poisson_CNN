import tensorflow as tf
import string
import math

@tf.function(experimental_relax_shapes=True)
def generate_smooth_function(ndims,grid_size,coefficients_or_coefficients_size, homogeneous_bc = False, homogeneous_neumann_bc = False, return_coefficients = False, normalize = False, coefficients_return_shape = None):
    '''
    Generates a smooth function sampled on a grid the size of which is given by grid_size.

    Inputs:
    -grid_size: list of ints. determines the shape of the output
    -coefficients_or_coefficients_size: float tf.Tensor or list of ints. determines the number of Fourier coefficients to use per spatial dim if a list of ints, or the coefficients themselves if a float tf.Tensor. larger values result in less smooth functions
    -homogeneous_bc: bool. If set to true, the value of the result on the boundaries will be 0.0 - i.e. only sine components will be used. TODO: rename to homogeneous_dirichlet_bc
    -homogeneous_neumann_bc: bool. If set to true, the derivative normal to the boundary will be 0.0 - i.e. only cosine components will be used. TODO: make the implementation cleaner
    -return_coefficients: bool. If set to true, the coefficients used to generate the function will be returned.
    -normalize: bool. If set to true, set max magnitude of the result to 1
    -coefficients_return_shape: tf.Tensor. If provided, and return_coefficients is set to True, the returned coefficients will be padded with 0s to this shape, as though additional Fourier modes with 0 amplitudes exist. Useful for shape compatibility of returned coefficients when working with a batch of results returned by this function. 

    Outputs:
    -result: tf.Tensor. the generated smooth function.
    -coefficients: tf.Tensor of shape (2,) + coefficients_size if homogeneous_bc is False, or tf.Tensor of shape coefficients_size if homogeneous_bc is True. Fourier coeffs used to generate result.
    '''
    
    #ndims = len(ndims)#tf.shape(tf.shape(grid_size))[0]
    
    if isinstance(coefficients_or_coefficients_size,tf.Tensor) and coefficients_or_coefficients_size.dtype == tf.keras.backend.floatx():
        if homogeneous_bc:
            sin_coefficients = coefficients_or_coefficients_size
        elif homogeneous_neumann_bc:
            sin_coefficients = tf.zeros(tf.shape(coefficients_or_coefficients_size), coefficients_or_coefficients_size.dtype)
            cos_coefficients = coefficients_or_coefficients_size
        else:
            sin_coefficients = coefficients_or_coefficients_size[0]
            cos_coefficients = coefficients_or_coefficients_size[1]
    else:
        if homogeneous_neumann_bc:
            sin_coefficients = 2*tf.zeros(coefficients_or_coefficients_size,dtype=tf.keras.backend.floatx())
        else:
            sin_coefficients = 2*tf.random.uniform(coefficients_or_coefficients_size,dtype=tf.keras.backend.floatx())-1
        if not homogeneous_bc:
            cos_coefficients = 2*tf.random.uniform(coefficients_or_coefficients_size,dtype=tf.keras.backend.floatx())-1

    coefficients_size = tf.shape(sin_coefficients)

    coords = [tf.linspace(tf.constant(0.0,dtype=tf.keras.backend.floatx()),tf.constant(math.pi,dtype=tf.keras.backend.floatx()),grid_size[k]) for k in range(ndims)]
    wavenumbers = [tf.cast(tf.range(1,coefficients_size[k]+1),tf.keras.backend.floatx()) for k in range(ndims)]
    
    trig_arguments = [tf.einsum('i,j->ij',coord,wavenumber) for coord,wavenumber in zip(coords,wavenumbers)]
    #trig_values = [tf.sin(x)+tf.cos(x) if not homogeneous_bc else tf.sin(x) for x in trig_arguments]
    
    

    einsum_str = [string.ascii_uppercase[:ndims]] + [string.ascii_lowercase[k] + string.ascii_uppercase[k] for k in range(ndims)]
    einsum_str = ','.join(einsum_str) + '->' + string.ascii_lowercase[:ndims]

    sin_values = tf.einsum(einsum_str,sin_coefficients,*[tf.sin(x) for x in trig_arguments])

    if homogeneous_bc:
        res = sin_values
    else:
        cos_values = tf.einsum(einsum_str,cos_coefficients,*[tf.cos(x) for x in trig_arguments])
        res = sin_values + cos_values
    
    #res = tf.einsum(einsum_str,coefficients,*trig_values)
    
    if normalize:
        res = res/tf.reduce_max(tf.abs(res))

    if return_coefficients:
        if coefficients_return_shape is not None:
            paddings = tf.stack([tf.zeros((ndims,),dtype=tf.int32),coefficients_return_shape - tf.shape(sin_coefficients)],1)
            sin_coefficients = tf.pad(sin_coefficients,paddings,"CONSTANT",constant_values = 0.0)
            if homogeneous_bc:
                return res, sin_coefficients
            else:
                cos_coefficients = tf.pad(cos_coefficients,paddings,"CONSTANT",constant_values = 0.0)
                return res, tf.stack([sin_coefficients,cos_coefficients],0)
        else:
            if homogeneous_bc:
                return res, sin_coefficients
            else:
                return res, tf.stack([sin_coefficients,cos_coefficients],0)
    else:
        return res
