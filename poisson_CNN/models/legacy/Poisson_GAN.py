import tensorflow as tf
import copy

from .Poisson_CNN import Poisson_CNN
from .custom_blocks import ResnetBlock

from ..layers import SpatialPyramidPool
from ..dataset.generators.numerical import numerical_dataset_generator as ndg

class Poisson_Discriminator(tf.keras.models.Model):
    def __init__(self, data_format = 'channels_first', n_convstages = 3, conv_params = {'filters': 6, 'kernel_size': 13, 'activation': tf.nn.leaky_relu, 'padding': 'same', 'strides': 1}, resnetblock_params = {'kernel_size' : 13, 'activation' : tf.nn.leaky_relu}, spp_levels = [[3,3],4,6,8,12], dropout_fraction = 0.2, **kwargs):
        super().__init__(**kwargs)
        conv_params = copy.deepcopy(conv_params)
        self.data_format = data_format
        self.n_convstages = n_convstages
        if self.data_format == 'channels_first':
            batchnorm_axis = 1
        else:
            batchnorm_axis = -1
            
        stridedconv_filters = conv_params.pop('filters')
        stridedconv_filters = [stridedconv_filters*(2**(k)) for k in range(n_convstages)]
        self.reshaping_convs = [tf.keras.layers.Conv2D(filters = stridedconv_filters[k], data_format = self.data_format, **conv_params) for k in range(n_convstages)]
        self.resnet_blocks = [ResnetBlock(data_format = self.data_format, filters = stridedconv_filters[k], **resnetblock_params) for k in range(n_convstages)]
        self.batchnorms = [tf.keras.layers.BatchNormalization(axis = batchnorm_axis) for k in range(2*n_convstages)]
        self.rhs_spp = SpatialPyramidPool(spp_levels, data_format = self.data_format)
        
        self.dense_0 = tf.keras.layers.Dense(256, activation = tf.nn.leaky_relu)
        self.dense_1 = tf.keras.layers.Dense(100, activation = tf.nn.leaky_relu)
        self.dense_2 = tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)
    def call(self, inp):
        out = self.batchnorms[1](self.resnet_blocks[0](self.batchnorms[0](self.reshaping_convs[0](inp))))
        for k in range(1,self.n_convstages):
            out = self.batchnorms[2*k+1](self.resnet_blocks[k](self.batchnorms[2*k](self.reshaping_convs[k](out))))
        out = self.rhs_spp(out)
        return self.dense_2(self.dense_1(self.dense_0(out)))

class combined_model(tf.keras.models.Model):
    def __init__(self, discriminator, generator):
        super().__init__()
        self.discriminator = discriminator
        self.discriminator.trainable = False
        self.generator = generator
    def call(self, inp):
        out = self.generator(inp)
        out = self.discriminator(out)
        return out#self.discriminator(self.generator(inp))
    
class Poisson_GAN():
    def __init__(self, Poisson_CNN_args = {}, data_format = 'channels_first', discriminator_weights = None, generator_weights = None, learning_rate = 1e-5):
        self.data_format = data_format
        self.optimizer =  tf.keras.optimizers.Adam(learning_rate = learning_rate)
        
        self.generator = Poisson_CNN(**Poisson_CNN_args)
        self.generator([tf.random.uniform((10,1,64,64), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1,64), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1,64), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1,64), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1,64), dtype = tf.keras.backend.floatx()), tf.random.uniform((10,1), dtype = tf.keras.backend.floatx())])
        if generator_weights is not None:
            self.generator.load_weights(generator_weights)
        self.generator.compile(loss = self.generator.integral_loss, optimizer = self.optimizer)
        self.generator.run_eagerly = True
        
        self.discriminator = Poisson_Discriminator()
        self.discriminator(tf.random.uniform((10,1,100,100), dtype = tf.keras.backend.floatx()))
        if discriminator_weights is not None:
            self.discriminator.load_weights(discriminator_weights)
        self.discriminator.compile(loss = 'binary_crossentropy', optimizer = self.optimizer)
        self.discriminator.run_eagerly = True
        
        # self.combined_model = tf.keras.models.Model(self.generator, self.discriminator)
        # self.combined_model = tf.keras.Sequential()
        # self.combined_model.add(self.generator)
        # self.combined_model.add(self.discriminator)
        self.combined_model = combined_model(self.discriminator, self.generator)
        self.combined_model.compile(loss = 'binary_crossentropy', optimizer = self.optimizer)
        self.combined_model.run_eagerly = True

    def train(self, epochs, dataset_generator = ndg(batch_size = 20, batches_per_epoch = 50, randomize_rhs_smoothness = True, randomize_boundary_smoothness = True, randomize_boundary_max_magnitudes = True, return_boundaries = True, return_dx = True, random_output_shape_range = [[256,384],[256,384]], random_dx_range = [0.005,0.02])):
        ###Implement weight saving!!!
        for epoch in range(epochs):
            '''
            Train discriminator
            '''
            inp, soln = dataset_generator.__getitem__(0)
            self.discriminator.trainable = True
            discriminator_loss_real = self.discriminator.train_on_batch(soln, tf.ones((soln.shape[0],1), dtype = tf.keras.backend.floatx()))
            generator_soln = self.generator.predict(inp)
            discriminator_loss_fake = self.discriminator.train_on_batch(generator_soln, tf.zeros((soln.shape[0],1), dtype = tf.keras.backend.floatx()))

            '''
            Train generator
            '''
            inp, soln = dataset_generator.__getitem__(0)
            self.combined_model.discriminator.trainable = False
            generator_loss = self.combined_model.train_on_batch(inp, tf.ones((soln.shape[0],1), dtype = tf.keras.backend.floatx()))

            '''
            Evaluate generator MSE
            '''
            generator_mse = tf.keras.losses.MSE(soln, self.combined_model.generator(inp))

            if epoch % 5 == 0:
                print('----Epoch ' + str(epoch) + '---- ')
                print('Discriminator loss on real inputs: ' + str(discriminator_loss_real))
                print('Discriminator loss on fake inputs: ' + str(discriminator_loss_fake))
                print('Combined model loss: ' + str(generator_loss))
                print('Generator MSE: ' + str(tf.reduce_mean(generator_mse).numpy()))
        
