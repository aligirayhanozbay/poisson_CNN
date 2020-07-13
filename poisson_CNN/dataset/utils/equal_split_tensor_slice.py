import tensorflow as tf
import string

from . import split_indices

@tf.function
def pad_mask_along_dimension(original_size,padded_size):
    return tf.concat([tf.ones((original_size),dtype=tf.bool),tf.zeros((padded_size - original_size,),dtype=tf.bool)],0)

@tf.function
def get_slice_from_innermost_dimensions(n_spatialdims,tensor, lower_corner, upper_corner, pad_inner_dims_to_size = None, return_pad_mask = False, pad_value = 0.0):
    input_shape = tf.shape(tensor)
    input_ndims = tf.shape(input_shape)[0]

    slice_ndims = tf.shape(lower_corner)[0]

    n_outer_dims = (input_ndims - slice_ndims)

    extent = tf.concat([input_shape[:n_outer_dims], upper_corner-lower_corner],0)
    lower_corner = tf.concat([tf.zeros((n_outer_dims,),dtype = lower_corner.dtype),lower_corner],0)
    
    res = tf.slice(tensor,lower_corner,extent)

    if pad_inner_dims_to_size is not None:
        paddings = tf.stack([tf.zeros((input_ndims,),dtype=tf.int32),tf.concat([tf.zeros((n_outer_dims,),dtype=tf.int32), pad_inner_dims_to_size - extent[n_outer_dims:]],0)],1)

        res = tf.pad(res,paddings,mode='CONSTANT',constant_values = pad_value)

        if return_pad_mask:
            pad_mask_components = [pad_mask_along_dimension(extent[n_outer_dims + k],pad_inner_dims_to_size[k]) for k in range(n_spatialdims)]
            pad_mask_components = [tf.reshape(pad_mask_components[k],[1 for _ in range(k)] + [-1] + [1 for _ in range(n_spatialdims-k-1)]) for k in range(n_spatialdims)]
            pad_mask = pad_mask_components.pop()
            for component in pad_mask_components:
                pad_mask = tf.logical_and(pad_mask,component)
            return res, pad_mask
        
    return res

@tf.function
def equal_split_tensor_slice(tensor, bin_index, n_bins, ndims, pad_to_equal_size = False, pad_value = 0.0, return_pad_mask = False):
    input_shape = tf.shape(tensor)
    total_subsections = tf.reduce_prod(n_bins)
    
    dim_sizes = input_shape[-ndims:]

    indices = [split_indices(dim_sizes[k],n_bins[k]) for k in range(ndims)]
    lower_corner = tf.stack([indices[k][bin_index[k]] for k in range(ndims)],0)
    upper_corner = tf.stack([indices[k][bin_index[k]+1] for k in range(ndims)],0)

    if pad_to_equal_size:
        pad_to_size = tf.stack([tf.reduce_max(indices[k][1:] - indices[k][:-1]) for k in range(ndims)],0)
    else:
        pad_to_size = None
    
    bin_values = get_slice_from_innermost_dimensions(ndims,tensor,lower_corner,upper_corner,pad_to_size,return_pad_mask,pad_value)
    
    return bin_values

if __name__ == '__main__':
    q = tf.ones((10,1,229,336,171))
    s = tf.constant(2.0)
    n_bins = tf.constant([5,4,3])
    bin_idx = tf.constant([4,2,2])
    bin_val, pm = equal_split_tensor_slice(q,bin_idx,n_bins = n_bins, ndims = 3,pad_to_equal_size = True, pad_value = 0.0, return_pad_mask = True)
    print(bin_val)
    print(pm)
    '''
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(q)
        tape.watch(s)
        #r = q*s
        bin_vals= equal_split_tensor_slice(q,bin_idx,n_bins = n_bins, ndims = 3)
    g = tape.gradient(bin_vals,q)
    print(tf.where(g != 0.0))
    '''
    
