import tensorflow as tf

from ..layers import metalearning_conv

class metalearning_resnet(tf.keras.models.Model):
    def __init__(self, previous_layer_filters, filters, kernel_size, use_batchnorm = False, batchnorm_trainable = True, **other_metalearning_conv_args):
        super().__init__()

        self.conv0 = metalearning_conv(previous_layer_filters, previous_layer_filters, kernel_size, padding = 'same', **other_metalearning_conv_args)
        self.conv1 = metalearning_conv(previous_layer_filters, previous_layer_filters, kernel_size, padding = 'same', **other_metalearning_conv_args)
        self.conv2 = metalearning_conv(previous_layer_filters, filters, kernel_size, padding = 'same', **other_metalearning_conv_args)

        self.use_batchnorm = use_batchnorm
        if self.use_batchnorm:
            try:
                if other_metalearning_conv_args['data_format'] == 'channels_first':
                    batchnorm_axis = 1
                else:
                    batchnorm_axis = -1
            except:
                batchnorm_axis = 1
            self.batchnorm0 = tf.keras.layers.BatchNormalization(axis = batchnorm_axis, trainable = batchnorm_trainable)
            self.batchnorm1 = tf.keras.layers.BatchNormalization(axis = batchnorm_axis, trainable = batchnorm_trainable)

    @tf.function
    def call(self, inp):
        out = self.conv0(inp)
        if self.use_batchnorm:
            out = self.batchnorm0(out)
        out = self.conv1([out,inp[1]])
        if self.use_batchnorm:
            out = self.batchnorm1(out)
        out = inp[0] + out
        out = self.conv2([out,inp[1]])
        return out
