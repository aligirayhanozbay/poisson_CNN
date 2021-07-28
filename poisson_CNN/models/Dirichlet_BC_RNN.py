import tensorflow as tf
from collections.abc import Iterable

from ..layers import Upsample
from ..dataset.utils import compute_domain_sizes

class Dirichlet_BC_RNN(tf.keras.models.Model):
    _rnn_types = {
        'lstm': tf.keras.layers.LSTM,
        'gru': tf.keras.layers.GRU
    }
    def __init__(self, units, activations = 'tanh', RNN_type = 'lstm', resize_method = 'bicubic', data_format = 'channels_first', **rnn_args):
        super().__init__()

        n_layers = len(units)
        self.data_format = data_format
        
        if (callable(activations)) or (isinstance(activations, str)):
            activations = [activations for _ in range(n_layers)]
        
        if isinstance(RNN_type, str):
            RNN_type = self._rnn_types[RNN_type.lower()]
        else:
            RNN_type = RNN_type

        self.RNN_layers = []
        for activation, layer_units in zip(activations, units):
            self.RNN_layers.append(
                RNN_type(layer_units, activation=activation, return_sequences = True, time_major = False, **rnn_args)
            )

        self.resampling_layer = Upsample(2, data_format = self.data_format, resize_method = resize_method)

    @tf.function
    def get_domain_shape(self,inpshape):
        return (inpshape[2:] if self.data_format == 'channels_first' else inpshape[1:-1])

    def call(self, inp):
        bc, dx, x_output_resolution = inp

        bc_shape = tf.shape(bc)
        domain_shape = tf.concat([tf.expand_dims(x_output_resolution,0),self.get_domain_shape(bc_shape)],0)
        domain_sizes = compute_domain_sizes(tf.concat([dx,dx],1), domain_shape)

        if self.data_format == 'channels_first':
            bc = tf.transpose(bc, [0,2,1])

        out = self.RNN_layers[0](bc)
        for layer in self.RNN_layers[1:]:
            out = layer(out)

        out = tf.expand_dims(out, 1 if self.data_format == 'channels_first' else -1)

        out = self.resampling_layer([out, domain_sizes, domain_shape])

        return out

    def train_step(self,data):

        inputs, ground_truth = data
        bc,dx = inputs
        x_output_resolution = tf.shape(ground_truth)[2 if self.data_format == 'channels_first' else 1]

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            pred = self([bc,dx,x_output_resolution])
            loss = self.loss_fn(y_true = ground_truth, y_pred = pred, rhs = tf.zeros(tf.shape(ground_truth),dtype=ground_truth.dtype), dx = tf.concat([dx,dx],1))
        grads = tape.gradient(loss,self.trainable_variables)

        self.optimizer.apply_gradients(zip(grads,self.trainable_variables))

        return {'loss': loss, 'mse': tf.reduce_mean((pred - ground_truth)**2), 'lr': self.optimizer.learning_rate}

    def compile(self, loss, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss
        

if __name__ == '__main__':
    inp = [tf.random.uniform((10,1,75)), tf.random.uniform((10,1)), 64]
    mod = Dirichlet_BC_RNN([100,100,100])
    out = mod(inp)
    print(out.shape)
            
            

        
    
