import tensorflow as tf

def get_pooling_method(pool_downsampling_method, ndims):
    pool_downsampling_method = pool_downsampling_method[0].upper() + pool_downsampling_method[1:].lower()
    pooling_layer_name = 'tf.keras.layers.' + pool_downsampling_method + 'Pooling' + str(ndims) + 'D'
    return eval(pooling_layer_name)
