import tensorflow as tf
import multiprocessing

@tf.function(experimental_relax_shapes=True)
def set_max_magnitude(arr, return_scaling_factors = False):
    '''
    Helper method to set the max magnitude in an array
    '''
    arr,max_magnitude = arr
        
    scaling_factor = max_magnitude/tf.reduce_max(tf.abs(arr))
    if return_scaling_factors:
        return arr * scaling_factor, scaling_factor
    else:
        return arr * scaling_factor
    
@tf.function(experimental_relax_shapes=True)
def set_max_magnitude_in_batch(arr, max_magnitude, return_scaling_factors = tf.constant(False)):
    '''
    Set max magnitudes within each batch
    '''
    if isinstance(max_magnitude, float):
        max_magnitude = tf.cast(tf.keras.backend.tile(tf.constant([max_magnitude]), [tf.shape(arr)[0]]), arr.dtype)
    elif tf.size(max_magnitude) == 1:
        max_magnitude = tf.keras.backend.tile(tf.reshape(max_magnitude,[1]), [tf.shape(arr)[0]])

    if return_scaling_factors == True:
        return_dtype = (arr.dtype, arr.dtype)
    else:
        return_dtype = arr.dtype

    return tf.map_fn(lambda x: set_max_magnitude(x, return_scaling_factors = return_scaling_factors), (arr, max_magnitude), dtype = return_dtype)
