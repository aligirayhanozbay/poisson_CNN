import tensorflow as tf

def apply_advanced_padding_and_call_conv_layer(padding_mode, conv_layer, constant_padding_value = 0.0):
    if conv_layer.padding == 'same':
        conv_layer.padding = 'valid'
    padding_mode = padding_mode.upper()
    data_format = conv_layer.data_format
    kernel_shape = tf.constant(conv_layer.kernel_size)
    left_paddings = kernel_shape // 2
    right_paddings = kernel_shape // 2 - (1 - kernel_shape%2)
    paddings = tf.stack([left_paddings, right_paddings],-1)
    if data_format == 'channels_first':
        paddings = tf.concat([[[0,0],[0,0]],paddings],0)
    else:
        paddings = tf.concat([[[0,0]],paddings,[[0,0]]],0)
    @tf.function
    def pad_and_apply_convolution(x):
        out = tf.pad(x,paddings,mode=padding_mode, constant_values=constant_padding_value)
        out = conv_layer(out)
        return out
    return pad_and_apply_convolution
