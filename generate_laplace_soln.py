import numpy as np
import tensorflow as tf
from generate_cholesky_soln import cholesky_poisson_solve, poisson_matrix
from Boundary import Boundary1D

def generate_random_boundaries(nx, ny, batch_size = 1, max_random_magnitude = 1.0, smoothness = None, nonzero_boundaries = ['left', 'right', 'bottom', 'top'], return_with_expanded_dims = False, data_format = 'channels_first'):
    boundary_lengths = {'left' : ny, 'right' : ny, 'top' : nx, 'bottom' : nx}
    if isinstance(smoothness, int):
        smoothness = {'left' : smoothness, 'right' : smoothness, 'top' : smoothness, 'bottom' : smoothness}
    elif smoothness == None:
        smoothness = {'left' : np.random.randint(5,int(ny//1.5)), 'right' : np.random.randint(5,int(ny//1.5)), 'top' : np.random.randint(5,int(nx//1.5)), 'bottom' : np.random.randint(5,int(nx//1.5))}
    boundaries = {}
    for boundary in smoothness.keys():
        if boundary in nonzero_boundaries:
            try:
                boundaries[boundary] = tf.Variable(tf.cast(tf.transpose(tf.squeeze(tf.image.resize_images(2*tf.random.uniform((smoothness[boundary],1,batch_size))-1,[boundary_lengths[boundary],1], method = tf.image.ResizeMethod.BICUBIC), axis = 1)), tf.keras.backend.floatx()))
            except:
                boundaries[boundary] = tf.Variable(tf.cast(tf.transpose(tf.squeeze(tf.compat.v1.image.resize_images(2*tf.random.uniform((smoothness[boundary],1,batch_size))-1,[boundary_lengths[boundary],1], method = tf.image.ResizeMethod.BICUBIC), axis = 1)), tf.keras.backend.floatx()))
            if max_random_magnitude != np.inf:
                for i in range(int(boundaries[boundary].shape[0])):
                    scaling_factor = max_random_magnitude/tf.reduce_max(tf.abs(boundaries[boundary][i,:]))
                    boundaries[boundary][i,:].assign(boundaries[boundary][i,:] * scaling_factor)
            boundaries[boundary] = boundaries[boundary].numpy()
        else:
            boundaries[boundary] = tf.zeros((batch_size, boundary_lengths[boundary]), dtype = tf.keras.backend.floatx())
        if return_with_expanded_dims:
            if data_format == 'channels_first':
                boundaries[boundary] = tf.expand_dims(boundaries[boundary], axis = 1)
            else:
                boundaries[boundary] = tf.expand_dims(boundaries[boundary], axis = 2)
    return boundaries

def generate_laplace_soln(batch_size = 1, nx = None, ny = None, dx = None, smoothness = None, nonzero_boundaries = ['left'], max_random_magnitude = 1.0):
    '''
    Generates 2D random solution - Dirichlet boundary condition pairs for the Laplace equation
    
    Inputs:
    -batch_size : # of problem-solution pairs to generate, integer
    -nx : x dir. mesh size, integer
    -ny : y dir. mesh size, integer
    -dx : grid spacing, float
    -nonzero_boundaries: list containing boundaries that should be nonzero. possible values it can contain are 'left', 'right', 'bottom' and  'top'
    -smoothness: controls how noisy the BCs are. lower values are smoother. can either be a dict containing entries which correspond to the sides in nonzero_boundaries with int values, or an int (in which case the same smoothness value is asssigned to every side). smoothness value associated with a side cannot exceed the no. of gridpoints on that side.
    '''
    if nx == None:
        nx = np.random.randint(32,128)
    if ny == None:
        ny = np.random.randint(32,128)
    if dx == None:
        dx = 0.1*(np.random.rand() + 0.01)
    
    boundaries = generate_random_boundaries(nx, ny, batch_size = batch_size, smoothness = smoothness, nonzero_boundaries = nonzero_boundaries, max_random_magnitude = max_random_magnitude, return_with_expanded_dims = True)
    print(boundaries['left'].shape)
    return boundaries, cholesky_poisson_solve(rhses = tf.zeros((batch_size,1,nx,ny), dtype = tf.keras.backend.floatx()), boundaries = boundaries, h = dx), dx