import numpy as np
import tensorflow as tf
from scipy.interpolate import RectBivariateSpline
from collections.abc import Iterable
from Boundary import Boundary1D
import itertools, h5py, os, sys, time
from multiprocessing import Pool as ThreadPool


def poisson_matrix(m,n):
    '''
    Generates the matrix A to express the Poisson equation in the form Ax=b for an m-by-n grid
    
    The matrix returned shall be (m-2)*(n-2)-by-(m-2)*(n-2) in size
    
    YOU MUST RESHAPE THE RESULT FROM (i.e. solution = inv(poisson_matrix) * right_hand_size) FROM (...,(m-2)*(n-2)) TO (...,m-2,n-2) BY USING FORTRAN COLUMN-MAJOR ORDERING!!!!!!!!
    '''
    m = m-2
    n = n-2
    
    D = np.zeros((m,m), dtype = np.float64)
    i,j = np.indices(D.shape)
    D[i==j] = 4.0
    D[i==j-1] = -1.0
    D[i==j+1] = -1.0
    
    S = -np.eye(D.shape[0], dtype = np.float64)
    
    P = np.zeros((m*n,m*n), dtype = np.float64)
    ind = np.arange(0,m*(n+1), m)
    
    for i in range(len(ind)-1):
        P[ind[i]:ind[i+1], ind[i]:ind[i+1]] = D
        try:
            P[ind[i+1]:ind[i+2], ind[i]:ind[i+1]] = S
        except:
            pass
        try:
            P[ind[i-1]:ind[i], ind[i]:ind[i+1]] = S
        except:
            pass
    return P


def generate_random_RHS(n, n_controlpts = None, n_outputpts = None, supersample_method = tf.image.ResizeMethod.BICUBIC, max_random_magnitude = np.inf):
    
    '''
    This function generates random smooth RHS 'functions' defined pointwise by creating a random number field defined in n_controlpts and then super-sampling it using tf.image.resize_images. Results are stacked across the 1st dimension.
    
    n: no. of random RHSes to generate
    n_controlpts: Iterable or int. No. of control pts of the spline along each dimension. Smaller values lead to 'smoother' results.
    n_outputpts: Iterable or int. No. of gridpoints in each direction of the output 
    supersample_method: Supersampling method for tf.image.resize_images. Use bicubic for smooth RHSes - bilinear or nearest neighbor NOT recommended
    max_random_magnitude: If the maximum absolute value of a generated random RHS exceeds this value, the entire thing is rescaled so the max abs. val. becomes this value
    '''
    
    
    if isinstance(n, Iterable):
        n_controlpts = n[1]
        n_outputpts = n[2]
        try:
            supersample_method = n[3]
        except:
            pass
        try:
            max_random_magnitude = n[4]
        except:
            pass
        n = n[0]
    try:
        n_controlpts = tf.ones(n_outputpts.shape, dtype = tf.int32)*n_controlpts
    except:
        n_controlpts = tf.ones(len(n_outputpts), dtype = tf.int32)*n_controlpts
    rand = 2*tf.random.uniform(list(n_controlpts) + [n], dtype = tf.keras.backend.floatx())-1
    try:
        rhs = tf.Variable(tf.cast(tf.transpose(tf.image.resize_images(rand, n_outputpts[-2:], method=supersample_method, align_corners=True), [2,0,1]), dtype=tf.keras.backend.floatx()))
    except:
        rhs = tf.Variable(tf.cast(tf.transpose(tf.compat.v1.image.resize_images(rand, n_outputpts[-2:], method=supersample_method, align_corners=True), [2,0,1]), dtype=tf.keras.backend.floatx()))

    if max_random_magnitude != np.inf:
        for i in range(int(rhs.shape[0])):
            scaling_factor = max_random_magnitude/tf.reduce_max(tf.abs(rhs[i,...]))
            rhs[i,...].assign(rhs[i,...] * scaling_factor)

    return rhs
    

def poisson_RHS(F, boundaries = None, h = None):
    '''
    Generates the RHS vector b of a discretized Poisson problem in the form Ax=b.
    h = grid spacing
    boundaries = dict containing entries 'top', 'bottom', 'right' and 'left' which correspond to the Dirichlet BCs at these boundaries. Each entry must be a vector of length m or n, where m and n are defined as in te function poisson_matrix
    F = an m by n matrix containing the RHS values of the Poisson equation
    
    (i.e. this function merely takes the BC information and the (structured) RHS array to provide the RHS for the matrix eq. form)
    '''
    #print(F)
    if isinstance(F, Iterable):
        boundaries = F[1]
        h = F[2]
        F = F[0]
    
    if isinstance(boundaries, dict):
        boundaries = itertools.repeat(boundaries, F.shape[0])
    if isinstance(h, float):
        h = itertools.repeat(h, F.shape[0])
    
    i = 0
    it = zip(boundaries,h)
    for i in range(F.shape[0]):
        boundary_set, dx = next(it)
        F[i] = -dx**2 * F[i]
        F[i,...,1:-1,1] += np.array(boundary_set['top'])[1:-1]
        F[i,...,1:-1,-2] += np.array(boundary_set['bottom'])[1:-1]
        F[i,...,1,1:-1] += np.array(boundary_set['left'])[1:-1]
        F[i,...,-2,1:-1] += np.array(boundary_set['right'])[1:-1]
        if i == 0:
            #print(F[i])
            i += 1
    #print(F[0,...])
    return F[...,1:-1,1:-1].reshape(list(F[...,1:-1,1:-1].shape[:-2]) + [np.prod(F[...,1:-1,1:-1].shape[-2:])], order = 'F') #Fortran reshape order important to preserve structure!
 
def generate_dataset(batch_size, n, h, boundaries, smoothness_levels = 1, max_random_magnitude = 1.0, initial_smoothness = 5):
    '''
    Generates Poisson equation RHS-solution pairs with 'random' RHS functions.
    
    n: shape of the outputs
    h: grid spacing of the outputs
    boundaries: boundary conditions of the outputs; see poisson_RHS documentation
    initial_smoothness and smoothness_levels: See documentation for __main__
    batch_size: No of random pairs to generate per smoothness level
    max_random_magnitude: See documentation for generate_random_RHS
    
    Outputs a tf.Tensor of shape (batch_size * smoothness_levels, n[0], n[1])
    '''
    F = tf.concat(list(map(generate_random_RHS, zip(itertools.repeat(batch_size, smoothness_levels), itertools.cycle(np.arange(initial_smoothness, initial_smoothness + smoothness_levels)), itertools.repeat((n[0],n[1])), itertools.repeat(tf.image.ResizeMethod.BICUBIC) ,itertools.repeat(max_random_magnitude)))), axis=0)
    #print('RHSes generated.')
    return tf.expand_dims(cholesky_poisson_solve(F, boundaries,h), axis = 1), tf.expand_dims(F, axis = 1)

def cholesky_poisson_solve(rhses, boundaries, h, system_matrix = None, system_matrix_is_decomposed = False):
    '''
    Solves the Poisson equation for the given RHSes.
    
    rhses: tf.Tensor representing the RHS functions of the Poisson equation, defined across the last 2 dimensions
    boundaries: boundary conditions of the outputs; see poisson_RHS documentation
    h: grid spacing of the outputs
    system_matrix: Poisson equation LHS matrix generated by poisson_matrix OR its Cholesky decomposition computed by tf.linalg.cholesky. If the Cholesky decomposition is supplied, set system_matrix_is_decomposed to True.
    system_matrix_is_decomposed: Flag to declare if a provided system_matrix is already Cholesky decomposed. Setting this to True when the system_matrix isn't decomposed will lead to undefined behaviour. Ignored if system_matrix is None.
    
    Outputs a tf.Tensor of identical shape to rhses.
    
    Note: Not tested if this function works on CPU.
    '''
    if system_matrix == None:
        system_matrix = poisson_matrix(int(rhses.shape[-2]), int(rhses.shape[-1]))
        system_matrix_is_decomposed = False
    #import pdb
    #pdb.set_trace()
    if not system_matrix_is_decomposed:
        system_matrix_chol = tf.expand_dims(tf.linalg.cholesky(system_matrix), axis=0)
    else:
        system_matrix_chol = system_matrix
    
    def chol(r):
        return tf.linalg.cholesky_solve(system_matrix_chol, tf.expand_dims(r, axis = 0))
    
    try:
        #@tf.contrib.eager.defun
        def chol_solve(rhs_arr):
            return tf.map_fn(chol, rhs_arr)
    except:
        #@tf.function
        def chol_solve(rhs_arr):
            return tf.map_fn(chol, rhs_arr)
    #print(rhses.shape)
    try:
        rhs_vectors = tf.expand_dims(tf.transpose(tf.squeeze(poisson_RHS([np.array(rhses), boundaries, h])), (0,1)),axis=2)
    except:
        rhs_vectors = tf.expand_dims(tf.expand_dims(tf.squeeze(poisson_RHS([np.array(rhses), boundaries, h])),axis=1), axis=0)
    
    z = tf.reshape(chol_solve(rhs_vectors), list(rhses.shape[:-2]) + [int(rhses.shape[-1])-2, int(rhses.shape[-2])-2])
    z = tf.transpose(z, list(range(len(z.shape[:-2]))) + [len(z.shape)-1, len(z.shape)-2])
    
    soln = np.zeros(rhses.shape, dtype = np.float64)
    soln[...,:,0] = boundaries['top']
    soln[...,:,-1] = boundaries['bottom']
    soln[...,0,:] = boundaries['left']
    soln[...,-1,:] = boundaries['right']
    soln[...,1:-1,1:-1] = z
    
    return soln         

if __name__ == '__main__':
    opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.925)
    conf = tf.ConfigProto(gpu_options=opts)
    tf.enable_eager_execution(config=conf)
    tf.keras.backend.set_floatx('float64')
    
    import argparse
    #_, outputpath, ntest, h, batch_size, n_batches = sys.argv
    parser = argparse.ArgumentParser(description = "Generate a series of Poisson equation RHS-solution pairs with specified Dirichlet boundary conditions on rectangular domains")
    parser.add_argument('-o', help = "Path to output file", required = True)
    parser.add_argument('-n', help = "No of gridpoints per side. First integer provided will set the value for the horizontal direction and the second the vertical. If only 1 value is given, a square domain is assumed.", required = True)
    parser.add_argument('-dx', help = "Grid spacing", required = True)
    parser.add_argument('-s', '--smoothness', help = 'Smoothness level. A higher number will generate noisier RHSes.', required = False, default = 5)
    parser.add_argument('-sl', '--smoothness_levels', help = "Number of smoothness levels. For example, setting S to 5 and SL to 3 will create 3 sets of solutions with smoothness levels 5,6,7.", required = False, default = 20)
    parser.add_argument('-bs', '--batch_size' ,help = "No of solutions to generate per smoothness level", required = True)
    args = parser.parse_args()

    m,n = [int(arg) for arg in args.n.split(' ')]
    dx = float(args.dx)
    batch_size = int(args.batch_size)
    smoothness = int(args.smoothness)
    smoothness_levels = int(args.smoothness_levels)
    outputpath = str(args.o)
    
    folder = 'dataset_' + str(ntest)
    boundary_top = Boundary1D('Dirichlet', [(0,n*dx),(m*dx,n*dx)], orientation='clockwise', RHS_function=lambda t: t-t, boundary_rhs_is_parametric=True)
    boundary_right = Boundary1D('Dirichlet', [(m*dx,n*dx),(m*dx,0)], orientation='clockwise', RHS_function=lambda t: t-t, boundary_rhs_is_parametric=True)
    boundary_bottom = Boundary1D('Dirichlet', [(m*dx,0),(0,0)], orientation='clockwise', RHS_function=lambda t: t-t, boundary_rhs_is_parametric=True)
    boundary_left = Boundary1D('Dirichlet', [(0,0),(0,n*dx)], orientation='clockwise', RHS_function=lambda t: t-t, boundary_rhs_is_parametric=True)
    boundaries = {'top': boundary_top.RHS_evaluate(np.linspace(boundary_top.t.min(),boundary_top.t.max(),m)), 'right': boundary_left.RHS_evaluate(np.linspace(boundary_right.t.min(),boundary_right.t.max(),n)), 'bottom': boundary_bottom.RHS_evaluate(np.linspace(boundary_bottom.t.min(),boundary_bottom.t.max(),m)), 'left': boundary_left.RHS_evaluate(np.linspace(boundary_left.t.min(),boundary_left.t.max(),n))}

    t0 = time.time()
    soln,F = generate_dataset(batch_size=batch_size, n = ntest, h = dx, smoothness_levels=smoothness_levels, boundaries=boundaries, initial_smoothness = smoothness)
    t1 = time.time()
    print('Generation of training data took ' + str(t1-t0) + ' seconds')
    with h5py.File(outputpath, 'w') as hf:
        hf.create_dataset('soln', data=soln)
        hf.create_dataset('F', data=F)
    print('Data saved.')
    print('Max RHS  : ' + str(tf.reduce_max(F)))
    print('Min RHS  : ' + str(tf.reduce_min(F)))
    print('Max soln : ' + str(tf.reduce_max(soln)))
    print('Min soln : ' + str(tf.reduce_min(soln)))
