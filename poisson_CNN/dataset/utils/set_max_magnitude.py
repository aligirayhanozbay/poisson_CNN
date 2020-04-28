import tensorflow as tf

@tf.function
def set_max_magnitude(arr, max_magnitude = None, return_scaling_factors = False):
    '''
    Helper method to set the max magnitude in an array
    '''
    if max_magnitude == None:
        max_magnitude = arr[1]
        arr = arr[0]
        
    scaling_factor = max_magnitude/tf.reduce_max(tf.abs(arr))
    if return_scaling_factors:
        return arr * scaling_factor, scaling_factor
    else:
        return arr * scaling_factor
    
@tf.function
def set_max_magnitude_in_batch(arr, max_magnitude, return_scaling_factors = False):
    '''
    Set max magnitudes within each batch in parallel
    '''
    if isinstance(max_magnitude, float):
        #max_magnitude = tf.cast(tf.constant([max_magnitude for k in range(arr.shape[0])]), arr.dtype)
        max_magnitude = tf.cast(tf.keras.backend.tile(tf.constant([max_magnitude]), [tf.shape(arr)[0]]), arr.dtype)

    if return_scaling_factors:
        return_dtype = (arr.dtype, arr.dtype)
    else:
        return_dtype = arr.dtype

    return tf.map_fn(lambda x: set_max_magnitude(x, return_scaling_factors = return_scaling_factors), (arr, max_magnitude), dtype = return_dtype, parallel_iterations = multiprocessing.cpu_count())
