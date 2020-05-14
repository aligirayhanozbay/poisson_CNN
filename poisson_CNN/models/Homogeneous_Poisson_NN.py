import tensorflow as tf

from ..layers import metalearning_conv, metalearning_deconvupscale

class Homogeneous_Poisson_NN(tf.keras.models.Model):
    def __init__(self, ndims, data_format = 'channels_first', pre_pooling_convolutions = 5, post_pooling_convolutions = 5, initial_kernel_size = 19, final_kernel_size = 3, pooling_block_number = 6, max_filters = 32, use_bias = False, metalearning_conv_pre_output_dense_units = [8,16]):

        super().__init__()
        
        pre_pooling_convolution_filters = [1] + [int(max_filters*(k+1)/pre_pooling_convolutions) for k in range(pre_pooling_convolutions)]
        pre_pooling_convolution_kernel_size = initial_kernel_size
        self.pre_pooling_convolution_layers = [metalearning_conv(previous_layer_filters = pre_pooling_convolution_filters[k], filters = pre_pooling_convolution_filters[k+1], kernel_size = pre_pooling_convolution_kernel_size, padding = 'same', padding_mode = 'SYMMETRIC', data_format = data_format, conv_activation = tf.nn.leaky_relu, use_bias = use_bias, dimensions = ndims, dense_activations = tf.nn.leaky_relu, pre_output_dense_units = metalearning_conv_pre_output_dense_units) for k in range(pre_pooling_convolutions)]


        
