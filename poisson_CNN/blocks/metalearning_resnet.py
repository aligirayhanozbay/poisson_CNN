import tensorflow as tf

from ..layers import metalearning_conv

class metalearning_resnet(tf.keras.models.Model):
    def __init__(self, previous_layer_filters, filters, kernel_size, **other_metalearning_conv_args):
        super().__init__()

        self.conv0 = metalearning_conv(previous_layer_filters, filters, kernel_size, padding = 'same', **other_metalearning_conv_args)
        self.conv1 = metalearning_conv(previous_layer_filters, filters, kernel_size, padding = 'same', **other_metalearning_conv_args)

    @tf.function
    def call(self, inp):
        return inp[0] + self.conv1([self.conv0(inp),inp[1]])
