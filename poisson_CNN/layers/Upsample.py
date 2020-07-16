import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections.abc import Iterable

@tf.function
def meshgrid_of_single_domain_size_set(domain_sizes,npts,ndims):
    linspace_start = tf.constant(0.0,tf.keras.backend.floatx())
    coords = [tf.linspace(linspace_start,domain_sizes[k], num = npts[k]) for k in range(ndims)]
    mg = tf.stack(tf.meshgrid(*coords,indexing='ij'),-1)
    mg = tf.reshape(mg, [-1,ndims])
    return mg

class Upsample(tf.keras.layers.Layer):
    def __init__(self, ndims, data_format = 'channels_first', resize_method = 'bilinear'):
        '''
        Init arguments:
        -ndims: int. Number of spatial dimensions in the inputs
        -data_format: string. 'channels_first' or 'channels_last'.
        -resize_method: string or tf.image.ResizeMethod. See tf.image.resize documentation. Used only for 2d images, any other # of dimensions use multilinear.
        '''
        super().__init__()
        self.data_format = data_format
        self.ndims = ndims
        self.resize_method = resize_method
        
        if self.ndims != 2 and resize_method != 'bilinear':
            import warnings
            warnings.warn('Upsample - Currently only multilinear upsampling is supported for ndims other than 2')

    @tf.function
    def call(self, inputs):
        inp, domain_sizes, output_shape = inputs

        input_shape = tf.shape(inp)
        batch_size = input_shape[0]
        n_channels = input_shape[1]

        if self.ndims != 2:
            if self.data_format == 'channels_last':
                inp = tf.einsum('i...j->ij...', inp)

            lower_domain_coordinates = tf.zeros(tf.shape(domain_sizes),dtype=domain_sizes.dtype)

            output_coords = tf.map_fn(lambda x: meshgrid_of_single_domain_size_set(x, output_shape, self.ndims), domain_sizes, dtype = inp.dtype)

            output_coords = tf.expand_dims(output_coords,1)
            lower_domain_coordinates = tf.expand_dims(lower_domain_coordinates,1)
            domain_sizes = tf.expand_dims(domain_sizes,1)
            out = tfp.math.batch_interp_regular_nd_grid(output_coords, lower_domain_coordinates, domain_sizes, inp, -self.ndims)
            output_shape = tf.concat([[batch_size],[n_channels],output_shape],0)

            out = tf.reshape(out, output_shape)
        else:
            if self.data_format == 'channels_first':
                inp = tf.einsum('ij...->i...j', inp)
            out = tf.cast(tf.image.resize(inp, output_shape, method = self.resize_method, preserve_aspect_ratio = False, antialias = False),tf.keras.backend.floatx())
            if self.data_format == 'channels_first':
                out = tf.einsum('i...j->ij...', out)
        
        return out

if __name__ == '__main__':
    inp = tf.random.uniform((10,1,250,250))
    domain_sizes = tf.random.uniform((10,2))
    output_shape = tf.constant([375,375])

    lay = Upsample(ndims = 2)

    print(lay([inp, domain_sizes, output_shape]))

    
        
        
