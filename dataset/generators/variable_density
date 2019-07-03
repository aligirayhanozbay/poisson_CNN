import numpy as np
import copy
import tensorflow as tf
from scipy.interpolate import RectBivariateSpline
from collections.abc import Iterable
from Boundary import Boundary1D
import itertools, h5py, os, sys, time
from multiprocessing import Pool as ThreadPool
from generate_cholesky_soln import generate_random_RHS#, poisson_RHS
from collections.abc import Iterator

def DOESNTWORK_poisson_matrix(rho, dx, dy = None):
    '''
    Generates the matrix A to express the Poisson equation in the form Ax=b for an m-by-n grid for the compressible problem:

    div 1/rho grad u = f
    
    The matrix returned shall be (m-2)*(n-2)-by-(m-2)*(n-2) in size

    This function can be also used with rho = np.ones((m,n)) to generate the Poisson matrix for an incompressible problem but with a stretched mesh
    
    YOU MUST RESHAPE THE RESULT FROM (i.e. solution = inv(poisson_matrix) * right_hand_size) FROM (...,(m-2)*(n-2)) TO (...,m-2,n-2) BY USING FORTRAN COLUMN-MAJOR ORDERING!!!!!!!!
    '''
    
    if not dy:
        dy = dx
    
    m = rho.shape[-2] - 2 #get shape, preallocate array
    n = rho.shape[-1] - 2
    P = np.zeros((rho.shape[0],m*n,m*n), dtype = np.float64)
    
    c1 = 2/(dx**2 * (rho[...,1:-1,1:-1] + rho[...,2:,1:-1])) #LHS matrix coefficients
    c2 = 2/(dx**2 * (rho[...,1:-1,1:-1] + rho[...,:-2,1:-1]))
    c3 = 2/(dy**2 * (rho[...,1:-1,1:-1] + rho[...,1:-1,2:]))
    c4 = 2/(dy**2 * (rho[...,1:-1,1:-1] + rho[...,1:-1,:-2]))
    c0 = -c1-c2-c3-c4
    
    ind = np.arange(0,m*(n+1), m)
    i,j = np.indices((m,m))
    for y in range(n):
        diagonal_block = np.zeros((rho.shape[0],m,m), dtype = np.float64)
        diagonal_block[:,i==j] = c0[:,:,y]
        diagonal_block[:,i==j+1] = c1[:,1:,y]
        diagonal_block[:,i==j-1] = c2[:,:-1,y]
        P[:,ind[y]:ind[y+1], ind[y]:ind[y+1]] = diagonal_block
    
    for y in range(n-1):
        superdiagonal_block = np.zeros((rho.shape[0],m,m), dtype = np.float64)
        superdiagonal_block[:,i==j] = c3[:,:,y]
        P[:,ind[y+1]:ind[y+2], ind[y]:ind[y+1]] = superdiagonal_block
        
    for y in range(1,n):
        subdiagonal_block = np.zeros((rho.shape[0],m,m), dtype = np.float64)
        subdiagonal_block[:,i==j] = c4[:,:,y]
        P[:,ind[y-1]:ind[y], ind[y]:ind[y+1]] = subdiagonal_block
        
    return P


def DOESNTWORK_poisson_matrix(rho, dx, dy = None):
    
    if not dy:
        dy = dx
    
    m = rho.shape[-2] - 2 #get shape, preallocate array
    n = rho.shape[-1] - 2
    P = np.zeros((rho.shape[0],m*n,m*n), dtype = np.float64)
    rhoinverse = 1/rho
    
    c_11 = -2.0 * (1/(dx**2) + 1/(dy**2)) * (rhoinverse[...,1:-1,1:-1])
    c_21 = 1/(dx**2) * (rhoinverse[...,1:-1,1:-1] + (rhoinverse[...,2:,1:-1] - rhoinverse[...,:-2,1:-1])/4)
    c_01 = 1/(dx**2) * (rhoinverse[...,1:-1,1:-1] + (-rhoinverse[...,2:,1:-1] + rhoinverse[...,:-2,1:-1])/4)
    c_12 = 1/(dy**2) * (rhoinverse[...,1:-1,1:-1] + (rhoinverse[...,1:-1,2:] - rhoinverse[...,1:-1,:-2])/4)
    c_10 = 1/(dy**2) * (rhoinverse[...,1:-1,1:-1] + (-rhoinverse[...,1:-1,2:] + rhoinverse[...,1:-1,:-2])/4)
    
    ind = np.arange(0,m*(n+1), m)
    i,j = np.indices((m,m))
    for y in range(n):
        diagonal_block = np.zeros((rho.shape[0],m,m), dtype = np.float64)
        diagonal_block[:,i==j] = c_11[:,:,y]
        diagonal_block[:,i==j+1] = c_21[:,1:,y]
        diagonal_block[:,i==j-1] = c_01[:,:-1,y]
        P[:,ind[y]:ind[y+1], ind[y]:ind[y+1]] = diagonal_block
    
    for y in range(n-1):
        superdiagonal_block = np.zeros((rho.shape[0],m,m), dtype = np.float64)
        superdiagonal_block[:,i==j] = c_12[:,:,y]
        P[:,ind[y+1]:ind[y+2], ind[y]:ind[y+1]] = superdiagonal_block
        
    for y in range(1,n):
        subdiagonal_block = np.zeros((rho.shape[0],m,m), dtype = np.float64)
        subdiagonal_block[:,i==j] = c_10[:,:,y]
        P[:,ind[y-1]:ind[y], ind[y]:ind[y+1]] = subdiagonal_block
        
    return P

def poisson_matrix(rho, dx, dy = None):
    
    if not dy:
        dy = dx
    
    m = rho.shape[-2] #get shape, preallocate array
    n = rho.shape[-1]
    P = np.zeros((rho.shape[0],m*n,m*n), dtype = np.float64)
    rhoinverse = 1/rho
    
    c_11 = -2.0 * (1/(dx**2) + 1/(dy**2)) * (rhoinverse[...,1:-1,1:-1])
    c_21 = 1/(dx**2) * (rhoinverse[...,1:-1,1:-1] + (rhoinverse[...,2:,1:-1] - rhoinverse[...,:-2,1:-1])/4)
    c_01 = 1/(dx**2) * (rhoinverse[...,1:-1,1:-1] + (-rhoinverse[...,2:,1:-1] + rhoinverse[...,:-2,1:-1])/4)
    c_12 = 1/(dy**2) * (rhoinverse[...,1:-1,1:-1] + (rhoinverse[...,1:-1,2:] - rhoinverse[...,1:-1,:-2])/4)
    c_10 = 1/(dy**2) * (rhoinverse[...,1:-1,1:-1] + (-rhoinverse[...,1:-1,2:] + rhoinverse[...,1:-1,:-2])/4)
    
    ind = np.arange(0,m*(n+1), m)
    i,j = np.indices((m,m))
    P[:,ind[0]:ind[1],ind[0]:ind[1]][:,i==j] = 1.0
    P[:,ind[-2]:ind[-1],ind[-2]:ind[-1]][:,i==j] = 1.0
    for y in range(1,n-1):
        diagonal_block = np.zeros((rho.shape[0],m,m), dtype = np.float64)
        diagonal_block[:,i==j] = tf.pad(c_11[:,:,y-1], tf.constant([[0,0],[1,1]]))
        diagonal_block[:,i==j+1] = tf.pad(c_21[:,:,y-1], tf.constant([[0,0],[0,1]]))
        diagonal_block[:,i==j-1] = tf.pad(c_01[:,:,y-1], tf.constant([[0,0],[1,0]]))
        diagonal_block[:,0,0] = 1.0
        diagonal_block[:,-1,-1] = 1.0
        P[:,ind[y]:ind[y+1], ind[y]:ind[y+1]] = diagonal_block
    
    for y in range(1,n-2):
        superdiagonal_block = np.zeros((rho.shape[0],m,m), dtype = np.float64)
        superdiagonal_block[:,i==j] = tf.pad(c_12[:,:,y-1], tf.constant([[0,0],[1,1]]))
        P[:,ind[y+1]:ind[y+2], ind[y]:ind[y+1]] = superdiagonal_block
        
    for y in range(2,n-1):
        subdiagonal_block = np.zeros((rho.shape[0],m,m), dtype = np.float64)
        subdiagonal_block[:,i==j] = tf.pad(c_10[:,:,y-1], tf.constant([[0,0],[1,1]]))
        P[:,ind[y-1]:ind[y], ind[y]:ind[y+1]] = subdiagonal_block
        
    return P
      
def compressible_poisson_solve(rho, rhses, boundaries, dx, dy = None, system_matrix = None):
    '''
    REWRITE THIS FUNCTION ENTIRELY TO MODIFY LHS MATRIX ONLY AND THEN CALL SOLVER FUNCTION INSTEAD!!!
    
    
    Solves the Poisson equation for the given RHSes.
    
    rhses: tf.Tensor representing the RHS functions of the Poisson equation, defined across the last 2 dimensions
    boundaries: boundary conditions of the outputs; see poisson_RHS documentation
    h: grid spacing of the outputs
    system_matrix: Poisson equation LHS matrix generated by poisson_matrix OR its Cholesky decomposition computed by tf.linalg.cholesky. If the Cholesky decomposition is supplied, set system_matrix_is_decomposed to True.
    system_matrix_is_decomposed: Flag to declare if a provided system_matrix is already Cholesky decomposed. Setting this to True when the system_matrix isn't decomposed will lead to undefined behaviour. Ignored if system_matrix is None.
    
    Outputs a tf.Tensor of identical shape to rhses.
    
    Note: Not tested if this function works on CPU.
    '''
    try: #handle spurious 1 dimensions
        rhses = tf.squeeze(rhses, axis = 1)
    except:
        pass
    boundaries = copy.deepcopy(boundaries)
    for boundary in boundaries.keys():
        try:
            boundaries[boundary] = tf.squeeze(boundaries[boundary], axis = 1)
        except:
            pass

    #generate poisson matrix, or use the provided one
    if system_matrix == None:
        system_matrix = tf.cast(poisson_matrix(rho, dx, dy = dy), tf.keras.backend.floatx())

    #put problem into Ax=b format
    try:
        rhs_vectors = tf.expand_dims(tf.transpose(tf.squeeze(poisson_RHS(np.array(rhses), boundaries = boundaries, rho = rho)), (0,1)),axis=2)
    except:
        rhs_vectors = tf.expand_dims(tf.expand_dims(tf.squeeze(poisson_RHS(np.array(rhses), boundaries = boundaries, rho = rho)),axis=1), axis=0)

    z = tf.reshape(tf.linalg.solve(system_matrix, rhs_vectors), list(rhses.shape[:-2]) + [int(rhses.shape[-1]), int(rhses.shape[-2])])
    z = tf.transpose(z, list(range(len(z.shape[:-2]))) + [len(z.shape)-1, len(z.shape)-2])

#     soln = np.zeros(rhses.shape, dtype = np.float64)
#     soln[...,:,0] = boundaries['top']
#     soln[...,:,-1] = boundaries['bottom']
#     soln[...,0,:] = boundaries['left']
#     soln[...,-1,:] = boundaries['right']
#     soln[...,1:-1,1:-1] = z

    return tf.expand_dims(z, axis = 1)

def generate_dataset(batch_size, n, boundaries, dx, dy = None, smoothness_levels = 1, max_random_magnitude = 1.0 - (1e-7), initial_smoothness = 5):
    smoothnesses = np.arange(initial_smoothness, initial_smoothness + smoothness_levels)
    F = tf.concat(list(map(generate_random_RHS, zip(itertools.repeat(batch_size, smoothness_levels), itertools.cycle(smoothnesses), itertools.repeat((n[0],n[1])), itertools.repeat(tf.image.ResizeMethod.BICUBIC) ,itertools.repeat(max_random_magnitude)))), axis=0)
    np.random.shuffle(smoothnesses)
    rho = 1e-7 + tf.abs(tf.concat(list(map(generate_random_RHS, zip(itertools.repeat(batch_size, smoothness_levels), itertools.cycle(smoothnesses), itertools.repeat((n[0],n[1])), itertools.repeat(tf.image.ResizeMethod.BICUBIC) ,itertools.repeat(max_random_magnitude)))), axis=0))

    return compressible_poisson_solve(rho, F, boundaries, dx, dy = dy), tf.stack([F, rho], axis = 1)
    
