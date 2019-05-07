import tensorflow as tf
from WeightedContractionLayer import WeightedContractionLayer
from Upsample import Upsample2
import itertools
import opt_einsum as oe
import numpy as np
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
    def __init__(self, data_format = 'channels_first', x_output_resolution = 256, n_sepconvblocks = 3, mae_component_weight = 0.0, mse_component_weight = 0.0, n_quadpts= 20, Lp_norm_power = 2):
        super().__init__()
        self.mae_component_weight = mae_component_weight
        self.mse_component_weight = mse_component_weight
        self.n_quadpts = n_quadpts
        self.p = Lp_norm_power
        self.x_output_resolution = x_output_resolution
        self.data_format = data_format
        self.conv1d_0 = tf.keras.layers.Conv1D(filters=4, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=256, kernel_size=5, padding='same', activation=tf.nn.leaky_relu, data_format = data_format)
        
        if data_format == 'channels_first':
            sepconvkernelshape = (256,5)
        else:
            sepconvkernelshape = (5,256)
        self.sepconvblocks_3 = [SepConvBlock(data_format = data_format, separable_kernel_size = sepconvkernelshape, separable_filters = 10, nonsep_filters = 10) for i in range(n_sepconvblocks)]
        
        self.conv2d_4 = tf.keras.layers.Conv2D(filters = 8, kernel_size = 5, activation = tf.nn.leaky_relu, padding = 'same', data_format = data_format)
        
        self.output_upsample_4 = Upsample2([-1, -1], data_format = data_format , resize_method = tf.image.ResizeMethod.BICUBIC)
        
        self.conv2d_5 = tf.keras.layers.Conv2D(filters = 1, kernel_size = 9, activation = tf.tanh, padding = 'same', data_format = data_format)
        
        self.dx_dense_0 = tf.keras.layers.Dense(4, activation = tf.nn.relu)
        self.dx_dense_1 = tf.keras.layers.Dense(4, activation = tf.nn.relu)
        self.dx_dense_2 = tf.keras.layers.Dense(8, activation = tf.nn.softmax)
    def call(self, inputs):
        self.dx = inputs[1]
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
    
    def integral_loss(self, y_true, y_pred):
        try:
            if self.data_format == 'channels_first':
                c = 0.5* tf.concat([self.dx * int(y_true.shape[-2]), self.dx * int(y_true.shape[-1])],1)
                coords = np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-2]),np.linspace(-1, 1, y_true.shape[-1]),indexing = 'xy'), dtype = np.float64).transpose((1,2,0)) #coordinates of each grid pt in the domain
            else:
                c = 0.5 * tf.concat([self.dx * int(y_true.shape[-3]), self.dx * int(y_true.shape[-2])],1)
                coords = np.array(np.meshgrid(np.linspace(-1, 1, y_true.shape[-3]),np.linspace(-1, 1, y_true.shape[-2]),indexing = 'xy'), dtype = np.float64).transpose((1,2,0)) #coordinates of each grid pt in the domain
        except:
            return 0.0*(y_true - y_pred)
        image_coords = [coords[0,:,0], coords[:,1,1]] #x and y coordinates separately
        quadrature_x, quadrature_w = tuple([np.polynomial.legendre.leggauss(self.n_quadpts)[i].astype(np.float64) for i in range(2)])

        quadpts = tf.constant(np.array(np.meshgrid(quadrature_x,quadrature_x,indexing = 'xy')).transpose((1,2,0)),dtype = tf.float64)
        #quadweights = tf.reduce_prod(c)*tf.tensordot(tf.squeeze(quadrature_w),tf.squeeze(quadrature_w),axes = 0)
        indices = [[],[]] #indices between each quadrature point lies - indices[0] is in x-dir and indices[1] is in the y-dir
        quad_coords = [quadpts[0,:,0], quadpts[:,1,1]] #x and y coordinates of each quad pt respectively
        #find the indices of coords between which every quad. pt. lies
        for i in range(len(indices)):
            j=0
            #does not work if more than 2 quad pts fall within 1 cell - fix later
            while len(indices[i]) < quadpts.shape[0] and j<image_coords[i].shape[0]:
                try:
                    if abs(float(quad_coords[i][len(indices[i])] - image_coords[i][j])) == float(min(abs(quad_coords[i][len(indices[i])] - image_coords[i][j-1]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j+1]))):
                        if quad_coords[i][len(indices[i])] - image_coords[i][j] < 0:
                            indices[i].append((j-1,j))
                        else:
                            indices[i].append((j,j+1))
                except:
                    if abs(float(quad_coords[i][len(indices[i])] - image_coords[i][j])) == float(min(abs(quad_coords[i][len(indices[i])] - image_coords[i][j-1]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j]))):
                        indices[i].append((j-1,j))
                j+=1
        
        index_combinations = tf.Variable(tf.zeros((quadpts.shape[0], quadpts.shape[1], 4 , 2), dtype = tf.int32), trainable = False, dtype = tf.int32) #array storing the 4 index combinations on the original grid which surround each quad. pt.
        corners = tf.Variable(tf.zeros((quadpts.shape[0], quadpts.shape[1], 2 , 2), dtype = np.int32), dtype = tf.int32, trainable = False) #array storing the lower left corner and the upper right corner of each box stored in index_combinations. effectively this will contain [[xmin,ymin],[xmax,ymax]] for the rectangle around each quad pt.
        s=tf.constant(indices)
        for i in range(self.n_quadpts):
            for j in range(self.n_quadpts):
                index_combinations[i,j,:,:].assign(np.array(list(itertools.product(np.array(s)[0,i,:],np.array(s)[1,j,:]))))
        for i in range(self.n_quadpts):
            for j in range(self.n_quadpts):
                corners[i,j,:,:].assign(tf.stack([s[0,i,:],s[1,j,:]]))
        corners = tf.transpose(corners,(0,1,3,2))
        corner_coords = tf.gather_nd(tf.transpose(coords,(1,0,2)),corners)
        
        #compute the coefficients [b_11,b_12,b_21,b_22]
        #steps:
        #1. compute transpose(invert(array([[1,xmin,ymin,xmin*ymin],[1,xmin,ymax,xmin*ymax],[1,xmax,ymin,xmax*ymin],[1,xmax,ymax,xmax*ymax]]))) for the rectangle around each quad pt.
        #2. compute array([1,x_quadpt, y_quadpt, x_quadpt*y_quadpt]) for each quadpt
        #3. multiply the result of 1 and 2 for each quad pt.
        interpolation_matrix = tf.Variable(tf.ones((self.n_quadpts,self.n_quadpts,4,4), dtype = tf.keras.backend.floatx()), dtype = tf.keras.backend.floatx())
        interpolation_matrix[:,:,0:2,1].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,0:2,1],corner_coords[:,:,0,0], backend = 'tensorflow'))
        interpolation_matrix[:,:,2:,1].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,2:,1],corner_coords[:,:,1,0], backend = 'tensorflow'))
        interpolation_matrix[:,:,0::2,2].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,0::2,2],corner_coords[:,:,0,1], backend = 'tensorflow'))
        interpolation_matrix[:,:,1::2,2].assign(oe.contract('ijk,ij->ijk',interpolation_matrix[:,:,1::2,2], corner_coords[:,:,1,1], backend = 'tensorflow'))
        interpolation_matrix[:,:,:,3].assign(oe.contract('...,...,...->...',interpolation_matrix[:,:,:,3],interpolation_matrix[:,:,:,1], interpolation_matrix[:,:,:,2], backend = 'tensorflow'))
        interpolation_matrix = tf.transpose(tf.linalg.inv(interpolation_matrix), (0,1,3,2))
        q = tf.Variable(tf.ones((self.n_quadpts,self.n_quadpts,4), dtype = tf.keras.backend.floatx()), dtype = tf.keras.backend.floatx())
        q[:,:,1].assign(tf.transpose(quadpts[:,:,0]))
        q[:,:,2].assign(tf.transpose(quadpts[:,:,1]))
        q[:,:,3].assign(tf.multiply(q[:,:,1],q[:,:,2]))
        
        b = oe.contract('ijkl, ijl->ijk', interpolation_matrix, q, backend = 'tensorflow')


        quadweights = oe.contract('i,j,k->ijk',tf.squeeze(quadrature_w),tf.squeeze(quadrature_w), tf.reduce_prod(c,axis = 1), backend = 'tensorflow')
        if self.data_format == 'channels_first':
            interp_pts = tf.squeeze(tf.gather_nd(tf.transpose(y_true - y_pred, (2,3,1,0)), index_combinations), axis = 3)
        else:
            interp_pts = tf.squeeze(tf.gather_nd(tf.transpose(y_true - y_pred, (1,2,3,0)), index_combinations), axis = 3)

        # print('---data---')
        # print((y_true - y_pred).shape)
        # print('---index combinations---')
        # print(index_combinations.shape)
        # print('---interp_pts---')
        # print(interp_pts.shape)
        # print('---quadweights---')
        # print(quadweights[...,0])
        # print('---b---')
        # print(b.shape)

        values_at_quad_pts = oe.contract('ijkl, ijk->ijl', interp_pts, b, backend = 'tensorflow')

        loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(quadweights, values_at_quad_pts**self.p), axis = (0,1))**(1/self.p))
        # print('---values_at_quad_pts---')
        # print(values_at_quad_pts.shape)
        if self.mae_component_weight != 0.0:
            loss = loss + self.mae_component_weight * tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))
        if self.mse_component_weight != 0.0:
            loss = loss + self.mse_component_weight * tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
        return loss

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