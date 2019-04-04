import tensorflow as tf
from WeightedContractionLayer import WeightedContractionLayer
from Upsample import Upsample2
import itertools

class Dirichlet_BC_NN(tf.keras.models.Model):
    def __init__(self, pooling_layer_number = 6, resize_methods = None, data_format = 'channels_first', other_dim_output_resolution = 256):
        super().__init__()
        self.pooling_layer_number = pooling_layer_number
        self.data_format = data_format
        self.other_dim_output_resolution = other_dim_output_resolution
        if not resize_methods:
            try:
                self.resize_methods = [tf.compat.v1.image.ResizeMethod.BICUBIC for i in range(self.pooling_layer_number-2)] + [tf.compat.v1.image.ResizeMethod.BILINEAR, tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR]
            except:
                self.resize_methods = [tf.image.ResizeMethod.BICUBIC for i in range(self.pooling_layer_number-2)] + [tf.image.ResizeMethod.BILINEAR, tf.image.ResizeMethod.NEAREST_NEIGHBOR]
        else:
            self.resize_methods = resize_methods

        if data_format == 'channels_first':
            self.transpose_0 = tf.keras.layers.Permute((2,1))
        else:
            self.transpose_0 = lambda x: x
        self.conv1d_0 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = 'channels_last')
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = 'channels_last')
        self.pooling_layers_1 = [tf.keras.layers.AveragePooling1D(data_format='channels_last', pool_size=2**p) for p in range(1,1+self.pooling_layer_number)]
        
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation=tf.nn.leaky_relu, data_format = 'channels_last')
        self.convs_on_pooled_layers_2 = [tf.keras.layers.Conv1D(filters=256, kernel_size=3, padding='same', activation=tf.nn.leaky_relu, data_format = 'channels_last') for p in self.pooling_layers_1[:-2]] + [tf.keras.layers.Conv1D(filters=256, kernel_size=1, padding='same', activation=tf.nn.leaky_relu, data_format = 'channels_last') for p in self.pooling_layers_1[-2:]]
#        self.expand_dims_2 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 3))
        self.upsample_2 = [Upsample2([-1,256], data_format = 'channels_last', resize_method = p) for p in self.resize_methods]

        self.stack_3 = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis = 1))
        self.weighted_contract_3 = WeightedContractionLayer('j,j...->...')
        self.output_upsample = Upsample2([-1, -1], data_format = 'channels_last' , resize_method = tf.image.ResizeMethod.BICUBIC)
        self.transpose_3 = tf.keras.layers.Permute((3,2,1))
        self.conv2d_3 = tf.keras.layers.Conv2D(filters = 32, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = 'channels_first')

        self.conv2d_4 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, activation = 'linear', padding = 'same', data_format = 'channels_first')

    def call(self, inputs):
        if self.data_format == 'channels_first':
            self.input_length = inputs.shape[-1]
        else:
            self.input_length = inputs.shape[-2]
            
        out = self.transpose_0(inputs)
        out = self.conv1d_0(out)
        out = self.conv1d_1(out)
        
        pools = [pooling_layer(out) for pooling_layer in self.pooling_layers_1]
        pools = [tf.expand_dims(p, axis = 3) for p in self.map_to_layers(self.convs_on_pooled_layers_2,pools)]
        pools = self.map_to_layers(self.upsample_2, list(zip(pools, itertools.repeat([self.input_length], self.pooling_layer_number))))
        
        out = tf.expand_dims(self.conv1d_2(out), axis = 3)

        
        out = self.stack_3([out] + pools)
        out = self.weighted_contract_3(out)
        #print(out.shape)
        out = self.output_upsample([out, [self.input_length, self.other_dim_output_resolution]])
        out = self.transpose_3(out)
        #print(out.shape)
        return  self.conv2d_4(self.conv2d_3(out))
        



    def map_to_layers(self, layers, inputs):
        assert len(layers) == len(inputs), 'Number of layers supplied must be equivalent to the number of inputs!'
        return [layers[i](inputs[i]) for i in range(len(inputs))]
