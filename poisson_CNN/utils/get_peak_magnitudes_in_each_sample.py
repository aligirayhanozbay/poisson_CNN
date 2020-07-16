import tensorflow as tf

@tf.function
def get_peak_magnitudes_in_each_sample(batch):
    '''
    Obtains the maximum absolute value in each slice across the 1st dimension
    Inputs:
    -batch: tf.Tensor of 1 or more dimensions
    Output:
    A tf.Tensor of shape batch.shape[0]
    '''
    return tf.map_fn(lambda x: tf.reduce_max(tf.abs(x)), batch)
