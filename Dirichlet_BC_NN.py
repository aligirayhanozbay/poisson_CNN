import tensorflow as tf
from WeightedContractionLayer import WeightedContractionLayer
from Upsample import Upsample2
import itertools

'''
Model attempt 1

'''
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
        self.preupsample_conv_3_0 = tf.keras.layers.SeparableConv2D(2, kernel_size = (256,3), padding = 'same', activation = tf.nn.leaky_relu, data_format = 'channels_last')
        self.preupsample_conv_3_1 = tf.keras.layers.SeparableConv2D(4, kernel_size = (256,3), padding = 'same', activation = tf.nn.leaky_relu, data_format = 'channels_last')
        self.preupsample_conv_3_2 = tf.keras.layers.SeparableConv2D(8, kernel_size = (256,3), padding = 'same', activation = tf.nn.leaky_relu, data_format = 'channels_last')
        self.preupsample_conv_3_3 = tf.keras.layers.SeparableConv2D(16, kernel_size = (256,3), padding = 'same', activation = tf.nn.leaky_relu, data_format = 'channels_last')
        self.output_upsample = Upsample2([-1, -1], data_format = 'channels_last' , resize_method = tf.image.ResizeMethod.BICUBIC)
        self.transpose_3 = tf.keras.layers.Permute((3,2,1))
        self.conv2d_3 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 3, activation = tf.nn.leaky_relu, padding = 'same', data_format = 'channels_first')

        self.conv2d_4 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 3, activation = 'tanh', padding = 'same', data_format = 'channels_first')

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
        out = self.preupsample_conv_3_3(self.preupsample_conv_3_2(self.preupsample_conv_3_1(self.preupsample_conv_3_0(out))))
        out = self.output_upsample([out, [self.input_length, self.other_dim_output_resolution]])
        out = self.transpose_3(out)
        #print(out.shape)
        return  self.conv2d_4(self.conv2d_3(out))
        



    def map_to_layers(self, layers, inputs):
        assert len(layers) == len(inputs), 'Number of layers supplied must be equivalent to the number of inputs!'
        return [layers[i](inputs[i]) for i in range(len(inputs))]

'''
Model attempt 2

'''
class SepConvBlock(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', separable_kernel_size  = (5,256), nonsep_kernel_size = 5, separable_activation = tf.nn.leaky_relu, nonsep_activation = tf.nn.leaky_relu, separable_filters = 8, nonsep_filters = 4):
        super().__init__()
        self.separableconv2d = tf.keras.layers.SeparableConv2D(separable_filters, kernel_size = separable_kernel_size, padding = 'same', activation = separable_activation, data_format = data_format)
        self.conv2d = tf.keras.layers.Conv2D(filters = nonsep_filters, kernel_size = nonsep_kernel_size, activation = nonsep_activation, padding = 'same', data_format = data_format)
        
    def call(self, inp):
        return self.conv2d(self.separableconv2d(inp))

class Dirichlet_BC_NN_2(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', x_output_resolution = 256, n_sepconvblocks = 4):
        super().__init__()
        self.x_output_resolution = x_output_resolution
        self.data_format = data_format
        self.conv1d_0 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        if data_format == 'channels_first':
            sepconvkernelshape = (256,5)
        else:
            sepconvkernelshape = (5,256)
        self.sepconvblocks_3 = [SepConvBlock(data_format = data_format, separable_kernel_size = sepconvkernelshape) for i in range(n_sepconvblocks)]
        
        self.conv2d_4 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        
        self.output_upsample_4 = Upsample2([-1, -1], data_format = data_format , resize_method = tf.image.ResizeMethod.BICUBIC)
        
        self.conv2d_5 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 9, activation = tf.tanh, padding = 'same', data_format = data_format)

    def call(self, inputs):
        if self.data_format == 'channels_first':
            self.input_length = inputs.shape[-1]
        else:
            self.input_length = inputs.shape[-2]
        out = self.conv1d_2(self.conv1d_1(self.conv1d_0(inputs)))
        if self.data_format == 'channels_first':
            out = tf.expand_dims(out, axis = 1)
            newshape = [self.x_output_resolution, self.input_length]
        else:
            out = tf.expand_dims(out, axis = 3)
            newshape = [self.input_length, self.x_output_resolution]
            
        for scb in self.sepconvblocks_3:
            out = scb(out)
            
        out = self.output_upsample_4([self.conv2d_4(out), newshape])
        return self.conv2d_5(out)

#best candidate
class Dirichlet_BC_NN_2B(tf.keras.models.Model): #variant to include dx info
    def __init__(self, data_format = 'channels_first', x_output_resolution = 256, n_sepconvblocks = 3):
        super().__init__()
        self.x_output_resolution = x_output_resolution
        self.data_format = data_format
        self.conv1d_0 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        if data_format == 'channels_first':
            sepconvkernelshape = (256,5)
        else:
            sepconvkernelshape = (5,256)
        self.sepconvblocks_3 = [SepConvBlock(data_format = data_format, separable_kernel_size = sepconvkernelshape, separable_filters = 24, nonsep_filters = 24) for i in range(n_sepconvblocks)]
        
        self.conv2d_4 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        
        self.output_upsample_4 = Upsample2([-1, -1], data_format = data_format , resize_method = tf.image.ResizeMethod.BICUBIC)
        
        self.conv2d_5 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 9, activation = tf.tanh, padding = 'same', data_format = data_format)
        
        self.dx_dense_0 = tf.keras.layers.Dense(4, activation = tf.nn.relu)
        self.dx_dense_1 = tf.keras.layers.Dense(4, activation = tf.nn.relu)
        self.dx_dense_2 = tf.keras.layers.Dense(8, activation = tf.nn.softmax)
    def call(self, inputs):
        dx_res = 1/(1e-8 + 10 * self.dx_dense_2(self.dx_dense_1(self.dx_dense_0(inputs[1]))))
        
        if self.data_format == 'channels_first':
            self.input_length = inputs[0].shape[-1]
            contr_expr = 'ijk,ij->ijk'
        else:
            self.input_length = inputs[0].shape[-2]
            contr_expr = 'ikj,ij->ikj'
        out = self.conv1d_2(tf.einsum(contr_expr, self.conv1d_1(self.conv1d_0(inputs[0])), dx_res))
        
        if self.data_format == 'channels_first':
            out = tf.expand_dims(out, axis = 1)
            newshape = [self.x_output_resolution, self.input_length]
        else:
            out = tf.expand_dims(out, axis = 3)
            newshape = [self.input_length, self.x_output_resolution]
            
        for scb in self.sepconvblocks_3:
            out = scb(out)
            
        out = self.output_upsample_4([self.conv2d_4(out), newshape])
        return self.conv2d_5(out)

'''
Model attempt 3 - variational autoencoder type thing

'''
class Dirichlet_BC_NN_3(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', upsample_blocks = 4, x_output_resolution = 256):
        super().__init__()
        self.data_format = data_format
        self.x_output_resolution = x_output_resolution
        if self.data_format == 'channels_first':
            self.input_upsample = Upsample2([1,100], data_format = data_format)
        else:
            self.input_upsample = Upsample2([100,1], data_format = data_format)
        
        ##Encoder
        self.encoder_dense_0 = tf.keras.layers.Dense(75, activation = tf.nn.relu)
        self.encoder_dense_1 = tf.keras.layers.Dense(25, activation = tf.nn.relu)
        self.encoder_dense_2 = tf.keras.layers.Dense(15, activation = tf.nn.relu)
        
        #Decoder
        self.decoder_dense_0 = tf.keras.layers.Dense(1600, activation = tf.nn.leaky_relu)
        
        self.decoder_deconvs = [tf.keras.layers.Conv2DTranspose(filters = 8, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', strides = 2, data_format = data_format) for i in range(upsample_blocks)]
        self.decoder_separable_convs = [SepConvBlock(separable_kernel_size = (40,40), data_format = data_format) for i in range(upsample_blocks)]
        self.decoder_conv2d_4 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 11, activation = tf.tanh, padding = 'same', data_format = data_format)
        self.final_upsample = Upsample2([-1,-1], data_format = data_format)
    
    def call(self, inp):
        if self.data_format == 'channels_first':
            input_length = inp[0].shape[-1]
        else:
            input_length = inp[0].shape[-2]
        out = tf.squeeze(self.input_upsample(tf.expand_dims(inp[0], axis = 1)))
        out = tf.concat([self.encoder_dense_2(self.encoder_dense_1(self.encoder_dense_0(out))), inp[1]], axis = 1)
        
        if self.data_format == 'channels_first':
            out = tf.reshape(self.decoder_dense_0(out), (out.shape[0], 64, 5, 5))
        else:
            out = tf.reshape(self.decoder_dense_0(out), (out.shape[0], 5, 5, 64))
        
        for i in range(len(self.decoder_deconvs)):
            out = self.decoder_separable_convs[i](self.decoder_deconvs[i](out))
        return self.final_upsample([self.decoder_conv2d_4(out), [self.x_output_resolution, input_length]])

'''
Model attempt - direct prediction
'''
class Dirichlet_BC_NN_4(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', upsample_blocks = 4, x_output_resolution = 256):
        super().__init__()
        self.data_format = data_format
        self.x_output_resolution = x_output_resolution
        if self.data_format == 'channels_first':
            self.input_upsample = Upsample2([1,100], data_format = data_format)
        else:
            self.input_upsample = Upsample2([100,1], data_format = data_format)
            
        self.dense_0 = tf.keras.layers.Dense(500, activation = tf.nn.relu)
        self.dense_1 = tf.keras.layers.Dense(1000, activation = tf.nn.relu)
        self.dense_2 = tf.keras.layers.Dense(10000, activation = tf.nn.tanh)
        self.final_upsample = Upsample2([-1,-1], data_format = data_format)
        
    def call(self, inp):
        if self.data_format == 'channels_first':
            input_length = inp[0].shape[-1]
        else:
            input_length = inp[0].shape[-2]
        out = tf.squeeze(self.input_upsample(tf.expand_dims(inp[0], axis = 1)))
        out = self.dense_2(self.dense_1(self.dense_0(tf.concat([out, inp[1]], axis = 1))))
        return self.final_upsample([tf.reshape(out, (out.shape[0], 1, 100, 100)), [self.x_output_resolution, input_length]])