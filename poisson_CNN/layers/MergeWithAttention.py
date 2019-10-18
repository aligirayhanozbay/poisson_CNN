import tensorflow as tf
import numpy as np

class MergeWithAttention(tf.keras.layers.Add):
    '''
    Multiplies each input with a trainable weight \alpha_i and elementwise adds them.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.attention_weights = self.add_weight(name = 'attention_weights', shape = [len(input_shape), input_shape[0][1]], initializer='uniform', trainable=True)
        super().build(input_shape)
    
    #@tfe.defun
    def _merge_function(self, inputs):
        sm = tf.exp(self.attention_weights)/tf.reduce_sum(tf.exp(self.attention_weights))
        out = tf.einsum('j,ijkl->ijkl',sm[0], inputs[0])
        for i in range(1,len(inputs)):
            out += tf.einsum('j,ijkl->ijkl',sm[i], inputs[i])
        return out
        