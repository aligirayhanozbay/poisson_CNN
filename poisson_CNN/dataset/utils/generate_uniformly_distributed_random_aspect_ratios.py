import tensorflow as tf

@tf.function
def generate_uniformly_distributed_random_aspect_ratios(ndims,min_ar,max_ar,samples = 1,flip_probability = 0.5):
    '''
    Generate uniformly distributed random aspect ratios for a rectangular domain. Aspect ratios are calculated as L_x0/L_xk

    -ndims: int. Number of dimensions. E.g. ndims = 3 results in 2 aspect ratios being generated: Lx/Ly, Lx/Lz
    -min_ar: Float or float tf.Tensor. Minimum aspect ratio to be generated. This is the minimum value which max(L_x0/L_xk,L_xk/L_x0) can take. 
    -max_ar: Float of float tf.Tensor. Max aspect ratio to be generated. This is the max value which max(L_x0/L_xk,L_xk/L_x0) can take. 
    -samples: int. No of samples to generate.
    -flip_probability: float. Probability of generating results within the range [1/max_ar,1/min_ar] instead of [min_ar,max_ar].
    '''
    min_ar = tf.convert_to_tensor(min_ar,dtype=tf.keras.backend.floatx())
    max_ar = tf.convert_to_tensor(max_ar,dtype=tf.keras.backend.floatx())
    max_ar_inverse = 1/max_ar
    min_ar_inverse = 1/min_ar
    flip_mask = tf.cast(tf.math.round((flip_probability-0.5) + tf.random.uniform((samples,ndims-1))),tf.keras.backend.floatx())
    flipped_values = flip_mask * (tf.expand_dims(min_ar_inverse - max_ar_inverse,0)*tf.random.uniform(tf.shape(flip_mask),dtype=tf.keras.backend.floatx()) + max_ar_inverse)
    nonflipped_values = (1-flip_mask)* (tf.expand_dims(max_ar-min_ar,0)*tf.random.uniform(tf.shape(flip_mask),dtype=tf.keras.backend.floatx()) + min_ar)
    return flipped_values + nonflipped_values
