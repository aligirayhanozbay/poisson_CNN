import tensorflow as tf

def choose_conv_layer(ndims):
    return eval('tf.keras.layers.Conv' + str(ndims) + 'D')

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

def check_batchnorm_fused_enable():
    floatx = tf.keras.backend.floatx()
    batchnorm_fused_allowed_datatypes = ['float16', 'float32', 'bfloat16']
    if floatx in batchnorm_fused_allowed_datatypes:
        return True
    else:
        return False
    

class resnet(tf.keras.models.Model):
    def __init__(self, ndims, use_batchnorm = False, batchnorm_trainable = True, padding_mode = 'constant', constant_padding_value = 0.0, **conv_args):
        super().__init__()

        conv_layer = choose_conv_layer(ndims)

        self.conv_layers = [conv_layer(padding = 'valid',**conv_args) for _ in range(3)]

        self._apply_convolution = [apply_advanced_padding_and_call_conv_layer(padding_mode, layer, constant_padding_value = constant_padding_value) for layer in self.conv_layers]

        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            try:
                if conv_args['data_format'] == 'channels_first':
                    batchnorm_axis = 1
                else:
                    batchnorm_axis = -1
            except:
                batchnorm_axis = 1
            enable_fused_batchnorm = check_batchnorm_fused_enable()
            self.batchnorm0 = tf.keras.layers.BatchNormalization(axis = batchnorm_axis, trainable = batchnorm_trainable, fused = enable_fused_batchnorm)
            self.batchnorm1 = tf.keras.layers.BatchNormalization(axis = batchnorm_axis, trainable = batchnorm_trainable, fused = enable_fused_batchnorm)

    @tf.function
    def call(self, inp):
        out = self._apply_convolution[0](inp)
        if self.use_batchnorm:
            out = self.batchnorm0(out)
        out = self._apply_convolution[1](out)
        if self.use_batchnorm:
            out = self.batchnorm1(out)
        out = inp + out
        out = self._apply_convolution[2](out)
        return out

if __name__ == '__main__':
    inp = tf.random.uniform((10,3,100,100))
    mod = resnet(ndims = 2, use_batchnorm = True, batchnorm_trainable = True, padding_mode = 'symmetric', kernel_size = 7, filters = 3, kernel_initializer = 'zeros', use_bias = True, data_format = 'channels_first')
    out = mod(inp)
    import pdb
    pdb.set_trace()
