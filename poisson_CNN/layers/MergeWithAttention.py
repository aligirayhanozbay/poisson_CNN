import tensorflow as tf
import numpy as np

class MergeWithAttention(tf.keras.layers.Add):
    '''
    Multiplies each input with a trainable weight \alpha_i and elementwise adds them.
    '''
    def __init__(self, data_format = 'channels_first',n_channels = None, n_inputs = None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = data_format
        self.einsum_str = 'bc...n,nc->bc...' if self.data_format == 'channels_first' else 'b...cn,nc->b...c'
        if (n_channels is not None) and (n_inputs is not None):
            #datadim = 1 if self.data_format == 'channels_first' else -1
            #self.attention_weights = self.add_weight(name = 'attention_weights', shape = [n_inputs, n_channels], initializer='uniform', trainable=True)
            #self._already_initialized_weight = True
            single_tensor_shape = tf.TensorShape([10,n_channels])
            self.build([single_tensor_shape for _ in range(n_inputs)])
        elif bool(n_channels is None) ^ bool(n_inputs is None): #both of these have to be defined or neither should be defined
            raise(ValueError('Both n_channels and n_inputs must be None, or both must be an integer value'))
            

    def build(self, input_shape):
        #if not self._already_initialized_weight:
        self.built = True
        datadim = 1 if self.data_format == 'channels_first' else -1
        self.attention_weights = self.add_weight(name = 'attention_weights', shape = [len(input_shape), input_shape[0][datadim]], initializer='uniform', trainable=True)
        super().build(input_shape)
    
    @tf.function
    def _merge_function(self, inputs):
        sm = tf.exp(self.attention_weights)/tf.reduce_sum(tf.exp(self.attention_weights))
        inputs = tf.stack(inputs,-1)
        out = tf.einsum(self.einsum_str, inputs, sm)
        return out
        '''
        out = tf.einsum('j,ijkl->ijkl',sm[0], inputs[0])
        for i in range(1,len(inputs)):
            out += tf.einsum('j,ijkl->ijkl',sm[i], inputs[i])
        return out
        '''
        
