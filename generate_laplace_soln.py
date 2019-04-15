import numpy as np
import tensorflow as tf
from generate_cholesky_soln import cholesky_poisson_solve, poisson_matrix
from Boundary import Boundary1D

def generate_laplace_soln(batch_size = 1, nx = None, ny = None, dx = None, smoothness = None, nonzero_boundaries = ['left'], max_random_magnitude = 1.0):
    if nx == None:
        nx = np.random.randint(32,128)
    if ny == None:
        ny = np.random.randint(32,128)
    boundary_lengths = {'left' : ny, 'right' : ny, 'top' : nx, 'bottom' : nx}
    if dx == None:
        dx = 0.1*(np.random.rand() + 0.01)
    if isinstance(smoothness, int):
        smoothness = {'left' : smoothness, 'right' : smoothness, 'top' : smoothness, 'bottom' : smoothness}
    elif smoothness == None:
        smoothness = {'left' : np.random.randint(5,int(ny//1.5)), 'right' : np.random.randint(5,int(ny//1.5)), 'top' : np.random.randint(5,int(nx//1.5)), 'bottom' : np.random.randint(5,int(nx//1.5))}

    pm = tf.expand_dims(tf.linalg.cholesky(poisson_matrix(nx,ny)), axis = 0)
    boundaries = {}
    for boundary in smoothness.keys():
        if boundary in nonzero_boundaries:
            boundaries[boundary] = tf.Variable(tf.cast(tf.transpose(tf.squeeze(tf.image.resize_images(2*tf.random.uniform((smoothness[boundary],1,batch_size))-1,[boundary_lengths[boundary],1], method = tf.image.ResizeMethod.BICUBIC), axis = 1)), tf.keras.backend.floatx()))
            if max_random_magnitude != np.inf:
                for i in range(int(boundaries[boundary].shape[0])):
                    scaling_factor = max_random_magnitude/tf.reduce_max(tf.abs(boundaries[boundary][i,:]))
                    boundaries[boundary][i,:].assign(boundaries[boundary][i,:] * scaling_factor)
        else:
            boundaries[boundary] = tf.zeros((batch_size, boundary_lengths[boundary]), dtype = tf.keras.backend.floatx())

    return boundaries, tf.squeeze(tf.map_fn(lambda x: cholesky_poisson_solve(rhses = tf.zeros((1,1,nx,ny), dtype = tf.keras.backend.floatx()), boundaries = {'left': x[0], 'right': x[1], 'top': x[2], 'bottom': x[3]}, h = dx, system_matrix = pm, system_matrix_is_decomposed = True), list(boundaries.values()), (tf.float64)), axis = 1), dx