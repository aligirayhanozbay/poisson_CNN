import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model
import numpy as np

def conv_laplacian_loss(image_size, h):
    inp = Input((1,image_size[0], image_size[1]))
    laplacian_kernel = Conv2D(filters=1, kernel_size=5, activation='linear', data_format='channels_first', padding = 'same')(inp)
    mod = Model(inp,laplacian_kernel)
    w = np.zeros((5,5,1,1), dtype = np.float64)
    w[2,0,0,0] = -1
    w[0,2,0,0] = -1
    w[1,2,0,0] = 16.0
    w[2,1,0,0] = 16.0
    w[2,2,0,0] = -60
    w[3,2,0,0] = 16.0
    w[2,3,0,0] = 16.0
    w[2,4,0,0] = -1
    w[4,2,0,0] = -1
    mod.set_weights([(1/(12*h**2))*tf.constant(w, dtype=tf.float64),tf.constant([0.0], dtype=tf.float64)])
    #pdb.set_trace()
    @tf.contrib.eager.defun
    def laplacian_loss(rhs, solution):
        return tf.reduce_sum((mod(solution)[:,:,2:-2,2:-2]-rhs[:,:,2:-2,2:-2])**2)/tf.cast(tf.reduce_prod(rhs[:,:,2:-2,2:-2].shape), rhs.dtype)
    return laplacian_loss