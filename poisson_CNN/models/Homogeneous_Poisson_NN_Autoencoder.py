import tensorflow as tf
import math
import string

from ..dataset.utils import compute_domain_sizes, set_max_magnitude_in_batch

class Homogeneous_Poisson_NN_AE_Decoder_Fourier(tf.keras.models.Model):
    def __init__(self, nmodes, data_format = 'channels_first', ndims = None, final_dense_activation = tf.keras.activations.linear, intermediate_dense_layer_units = None, intermediate_dense_layer_activations = tf.keras.activations.linear, use_layernorm = False, **further_dense_layer_options):
        super().__init__()
        
        if ndims is None:
            ndims = len(nmodes)
        self.ndims = ndims

        self.data_format = data_format
        
        self.solution_assembly_einsum_string = 'B' + ',B'.join(string.ascii_lowercase[:self.ndims]) + '->B' + string.ascii_lowercase[:self.ndims]

        if isinstance(nmodes,int):
            nmodes = [nmodes for _ in range(ndims)]
        self.nmodes = nmodes
        self.coefficient_indices = []
        for k in range(len(self.nmodes)):
            self.coefficient_indices.append((Ellipsis, slice(sum(self.nmodes[:k]),sum(self.nmodes[:k]) + self.nmodes[k])))
            
        if intermediate_dense_layer_units is None:
            intermediate_dense_layer_units = []
        if intermediate_dense_layer_activations is None:
            intermediate_dense_layer_activations = []
        elif callable(intermediate_dense_layer_activations):
            intermediate_dense_layer_activations = [intermediate_dense_layer_activations for _ in range(len(intermediate_dense_layer_units))]
            
        #self.dense_layers = [tf.keras.layers.Dense(units, activation = activation, **further_dense_layer_options) for units,activation in zip(intermediate_dense_layer_units, intermediate_dense_layer_activations)] + [tf.keras.layers.Dense(tf.reduce_sum(self.nmodes), activation = final_dense_activation, **further_dense_layer_options)]
        print(further_dense_layer_options)
        self.dense_layers = []
        for units,activation in zip(intermediate_dense_layer_units, intermediate_dense_layer_activations):
            self.dense_layers.append(tf.keras.layers.Dense(units, activation = activation, **further_dense_layer_options))
            if use_layernorm:
                self.dense_layers.append(tf.keras.layers.LayerNormalization())
        self.dense_layers.append(tf.keras.layers.Dense(tf.reduce_sum(self.nmodes), activation = final_dense_activation, **further_dense_layer_options))

    @tf.function
    def build_hpnn_solution_component_from_coefficients(self, coefficients, output_shape):
        n_coefficients = tf.shape(coefficients)[-1]
        coords = tf.tile(tf.expand_dims(tf.linspace(0.0,1.0,output_shape),0),[n_coefficients,1])
        sine_arguments = tf.einsum('i,ix->ix',math.pi*(tf.range(n_coefficients, dtype = tf.keras.backend.floatx())+1), coords)
        sine_values = tf.sin(sine_arguments)
        result = tf.einsum('bc,cx->bx', coefficients, sine_values)
        return result

    @tf.function
    def call(self,inp):
        inp, output_shape = inp
        dense_layer_result = self.dense_layers[0](inp)
        for layer in self.dense_layers[1:]:
            dense_layer_result = layer(dense_layer_result)
            
        coefficients_per_dimension = []
        for indices in self.coefficient_indices:
            coefficients_per_dimension.append(dense_layer_result[indices])

        npts_per_dimension = tf.unstack(output_shape)
            
        solution_components = []

        for npts, coefficients in zip(npts_per_dimension, coefficients_per_dimension):
            solution_components.append(self.build_hpnn_solution_component_from_coefficients(coefficients, npts))

        solution = tf.expand_dims(tf.einsum(self.solution_assembly_einsum_string, *solution_components), 1 if self.data_format == 'channels_first' else -1)

        return solution

class Homogeneous_Poisson_NN_AE_Decoder_Conv(Homogeneous_Poisson_NN_AE_Decoder_Fourier):
    def __init__(self, control_pts_per_dim, data_format = 'channels_first', ndims = None, final_dense_activation = tf.keras.activations.linear, intermediate_dense_layer_units = None, intermediate_dense_layer_activations = tf.keras.activations.linear, use_layernorm = False, further_dense_layer_options = None, use_resnet = False):
        if further_dense_layer_options is None:
            further_dense_layer_options = {}
        super().__init__(nmodes = final_dense_units, data_format = data_format, ndims = ndims, final_dense_activation = final_dense_activation, intermediate_dense_layer_units = intermediate_dense_layer_units, intermediate_dense_layer_activations = intermediate_dense_layer_activations, use_layernorm = use_layernorm, **further_dense_layer_options)

        
        
        

class Homogeneous_Poisson_NN_Autoencoder_2D(tf.keras.models.Model):
    def __init__(self, decoder_config, decoder_type = 'conv', resnet_weights = None):
        super().__init__()
        
        self.encoder = tf.keras.applications.ResNet50V2(include_top = False, weights = resnet_weights, pooling = 'avg')
        self.data_format = self.encoder.layers[2].data_format
        _ = decoder_config.pop('ndims', None)
        _ = decoder_config.pop('data_format', None)

        #decoder_class = 
        
        self.decoder = Homogeneous_Poisson_NN_AE_Decoder_Fourier(ndims = 2, data_format = self.data_format, **decoder_config)
        self.ndims = 2

    @tf.function
    def generate_position_embeddings(self, batch_size, domain_shape):
        pos_embeddings = tf.stack([tf.broadcast_to(tf.reshape(tf.cos(math.pi * tf.linspace(0.0,1.0,domain_shape[k])),[1 for _ in range(k)] + [-1] + [1 for _ in range(self.ndims-k-1)]), domain_shape) for k in range(self.ndims)],0 if self.data_format == 'channels_first' else -1)
        pos_embeddings = tf.expand_dims(pos_embeddings,0)
        pos_embeddings = tf.tile(pos_embeddings, [batch_size] + [1 for _ in range(self.ndims+1)])
        return pos_embeddings

    @tf.function
    def call(self, inp):
        
        rhses, dx = inp

        inp_shape = tf.shape(rhses)
        if self.data_format == 'channels_first':
            domain_shape = inp_shape[2:]
        else:
            domain_shape = inp_shape[1:-1]
        batch_size = inp_shape[0]
        domain_sizes = compute_domain_sizes(dx, domain_shape)
        max_domain_sizes = tf.reduce_max(domain_sizes,1)
        pos_embeddings = self.generate_position_embeddings(batch_size, domain_shape)

        conv_inp = tf.concat([rhses, pos_embeddings], 1 if self.data_format == 'channels_first' else -1)
        domain_info = tf.concat([dx/domain_sizes,tf.einsum('ij,i->ij',domain_sizes,1/max_domain_sizes)],1)

        embedding = self.encoder(conv_inp)

        decoder_inp = tf.concat([embedding, domain_info],1)

        output = self.decoder([decoder_inp, domain_shape])
        output = set_max_magnitude_in_batch(output, 1.0)
        return output

    def train_step(self,data):

        inputs, ground_truth = data

        rhses, dx = inputs

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            pred = self(inputs)
            loss = self.loss_fn(y_true=ground_truth,y_pred=pred,rhs=rhses,dx=dx)
        grads = tape.gradient(loss,self.trainable_variables)
                
        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        return {'loss' : loss, 'mse': tf.reduce_mean((pred - ground_truth)**2)}

    def compile(self, loss, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss


if __name__ == '__main__':
    #nmodes, data_format = 'channels_first', ndims = None, final_dense_activation = tf.keras.activations.linear, intermediate_dense_layer_units = None, intermediate_dense_layer_activations = tf.keras.activations.linear, use_layernorm = False, **further_dense_layer_option
    decoder_args = {
        "nmodes": 64,
        "final_dense_activation": tf.nn.tanh,
        "intermediate_dense_layer_units": [1000,500],
        "intermediate_dense_layer_activations": tf.nn.leaky_relu,
        "use_layernorm": True,
        "kernel_regularizer": tf.keras.regularizers.l2()
    }
    
    mod = Homogeneous_Poisson_NN_Autoencoder_2D(decoder_args)
    from ..losses import loss_wrapper
    physics_informed_loss_config = {
		"stencil_sizes":[5,5],
		"orders":2,
		"normalize":False
	    }
    integral_loss_config = {
		"n_quadpts":20,
		"Lp_norm_power":2
	    }
    mod.compile(loss = loss_wrapper(2,1.0,integral_loss_config,1.0,physics_informed_loss_config), optimizer = tf.keras.optimizers.Adam())
    import numpy as np
    class dummy_data_generator(tf.keras.utils.Sequence):
        def __init__(self):
            super().__init__()
        def __len__(self):
            return 500
        def __getitem__(self,idx=0):
            inshape = [10,1,200+int(6*(np.random.rand()-0.5)),200+int(6*(np.random.rand()-0.5))]
            ro = tf.random.uniform(inshape)
            return [ro, tf.random.uniform((inshape[0],2))], ro
    dg = dummy_data_generator()
    mod.fit(dg)
    
