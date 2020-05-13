import tensorflow as tf

from . import split_indices

@tf.function
def get_slice_from_innermost_dimensions(tensor, lower_corner, upper_corner):
    input_shape = tf.shape(tensor)
    input_ndims = tf.shape(input_shape)[0]

    slice_ndims = tf.shape(lower_corner)[0]

    n_outer_dims = (input_ndims - slice_ndims)

    upper_corner = tf.concat([input_shape[:n_outer_dims], upper_corner-lower_corner],0)
    lower_corner = tf.concat([tf.zeros((n_outer_dims,),dtype = lower_corner.dtype),lower_corner],0)
    
    return tf.slice(tensor,lower_corner,upper_corner)

@tf.function
def equal_split_tensor_slice(tensor, bin_index, n_bins, ndims = None):
    input_shape = tf.shape(tensor)
    total_subsections = tf.reduce_prod(n_bins)
    if ndims is None:
        ndims = tf.shape(input_shape)[0]
    
    dim_sizes = input_shape[-ndims:]

    indices = [split_indices(dim_sizes[k],n_bins[k]) for k in range(ndims)]
    lower_corner = tf.stack([indices[k][bin_index[k]] for k in range(ndims)],0)
    upper_corner = tf.stack([indices[k][bin_index[k]+1] for k in range(ndims)],0)
    
    bin_values = get_slice_from_innermost_dimensions(tensor,lower_corner,upper_corner)
    
    return bin_values

if __name__ == '__main__':
    q = tf.ones((10,1,229,336,171))
    s = tf.constant(2.0)
    n_bins = tf.constant([5,4,3])
    bin_idx = tf.constant([0,2,2])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(q)
        tape.watch(s)
        #r = q*s
        bin_vals= equal_split_tensor_slice(q,bin_idx,n_bins = n_bins, ndims = 3)
    g = tape.gradient(bin_vals,q)
    print(tf.where(g != 0.0))
    
