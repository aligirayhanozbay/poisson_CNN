import tensorflow as tf

@tf.function(experimental_relax_shapes=True)
def set_max_magnitude(arr):
    '''
    Helper method to set the max magnitude in an array
    '''
    arr,max_magnitude = arr
        
    scaling_factor = max_magnitude/tf.reduce_max(tf.abs(arr))
    return arr * scaling_factor
    
@tf.function(experimental_relax_shapes=True)
def set_max_magnitude_in_batch(arr, max_magnitude):
    '''
    Set max magnitudes within each batch
    '''
    if isinstance(max_magnitude, float):
        max_magnitude = tf.cast(tf.keras.backend.tile(tf.constant([max_magnitude]), [tf.shape(arr)[0]]), arr.dtype)
    elif tf.size(max_magnitude) == 1:
        max_magnitude = tf.keras.backend.tile(tf.reshape(max_magnitude,[1]), [tf.shape(arr)[0]])

    return_dtype = arr.dtype

    return tf.map_fn(set_max_magnitude, (arr, max_magnitude), dtype = return_dtype)

@tf.function(experimental_relax_shapes=True)
def set_max_magnitude_and_return_scaling_factors(arr):
    '''
    Helper method to set the max magnitude in an array
    '''
    arr,max_magnitude = arr
        
    scaling_factor = tf.cast(max_magnitude, tf.keras.backend.floatx())/tf.reduce_max(tf.abs(arr))
    return arr * scaling_factor, scaling_factor


@tf.function(experimental_relax_shapes=True)
def set_max_magnitude_in_batch_and_return_scaling_factors(arr, max_magnitude):
    '''
    Set max magnitudes within each batch
    '''
    if isinstance(max_magnitude, float):
        max_magnitude = tf.cast(tf.keras.backend.tile(tf.constant([max_magnitude]), [tf.shape(arr)[0]]), arr.dtype)
    elif tf.size(max_magnitude) == 1:
        max_magnitude = tf.keras.backend.tile(tf.reshape(max_magnitude,[1]), [tf.shape(arr)[0]])

    return_dtype = (arr.dtype, arr.dtype)

    return tf.map_fn(set_max_magnitude_and_return_scaling_factors, (arr, max_magnitude), dtype = return_dtype)
