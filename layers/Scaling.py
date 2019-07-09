import tensorflow as tf
import numpy as np

class Scaling(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.scaling_weight = self.add_weight(name = 'scaling_weight', shape = (), initializer=tf.initializers.ones, trainable = True)

    def call(self, inp):
        return inp * self.scaling_weight
