import tensorflow as tf

def choose_conv_layer(ndims):
    return eval('tf.keras.layers.Conv' + str(ndims) + 'D')

def choose_conv_method(ndims):
    return eval('tf.keras.backend.conv' + str(ndims) + 'd')
