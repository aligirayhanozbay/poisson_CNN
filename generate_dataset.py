import numpy as np
import tensorflow as tf
from scipy.interpolate import RectBivariateSpline
from Boundary import Boundary1D
import itertools, h5py, os, sys, time
from multiprocessing import Pool as ThreadPool
import argparse
opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.925)
conf = tf.ConfigProto(gpu_options=opts)
tfe.enable_eager_execution(config=conf)
tf.keras.backend.set_floatx('float64')

def poisson_matrix(m,n):
    '''
    Generates the matrix A to express the Poisson equation in the form Ax=b for an m-by-n grid
    
    Them matrix returned shall be (m-2)*(n-2)-by-(m-2)*(n-2) in size
    
    CURRENTLY ONLY WORKS FOR SQUARE DOMAINS!!!!
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

def generate_random_RHS(n, n_controlpts = None, n_outputpts = None, s = 5, domain = [0,1,0,1]):
    
    '''
    This function generates random smooth RHS 'functions' defined pointwise using bivariate splines. 
    n: no. of random RHSes to generate
    n_controlpts: no. of control pts of the spline. Smaller values lead to 'smoother' results
    n_outputpts: no. of gridpoints in each direction of the output 
    s: see parameter s in scipy.interpolate.RectBivariateSpline
    domain: [x_min, x_max, y_min, y_max]
    '''
    
    
    if isinstance(n, Iterable):
        n_controlpts = n[1]
        n_outputpts = n[2]
        try:
            s = n[3]
        except:
            pass
        try:
            domain = n[4]
        except:
            pass
        n = n[0]
    
    x = np.linspace(domain[0], domain[1], n_controlpts)
    y = np.linspace(domain[2], domain[3], n_controlpts)
    if n_controlpts != n_outputpts:
        x_out = np.linspace(domain[0], domain[1], n_outputpts)
        y_out = np.linspace(domain[2], domain[3], n_outputpts)
    else:
        x_out = x
        y_out = y
            
    out = []
    for i in range(n):
        spl = RectBivariateSpline(x,y,2*np.random.rand(len(x), len(y))-1,s=s) #modify 3rd argument to pick RHS. np.random.rand(len(x), len(y)) is for random. 2*(i/n)*np.ones((len(x), len(y)), dtype = tf.keras.backend.floatx())-1
        v = spl(x_out,y_out)
        out.append(2*(v-np.min(v))/(np.max(v)-np.min(v))-1)
    return np.array(out)
    

def poisson_RHS(F, boundaries = None, h = None):
    '''
    Generates the RHS vector b of a discretized Poisson problem in the form Ax=b.
    h = grid spacing
    boundaries = dict containing entries 'top', 'bottom', 'right' and 'left' which correspond to the Dirichlet BCs at these boundaries. Each entry must be a vector of length m or n, where m and n are defined as in te function poisson_matrix
    F = an m by n matrix containing the RHS values of the Poisson equation
    
    (i.e. this function merely takes the BC information and the array from generate_random_RHS to provide the RHS for the matrix eq. form)
    '''
    
    if isinstance(F, Iterable):
        boundaries = F[1]
        h = F[2]
        F = F[0]
    
    F = -h**2 * F
    F[...,1:-1,1] = F[...,1:-1,1] + np.array(boundaries['top'])[1:-1]
    F[...,1:-1,-2] = F[...,1:-1,-2] + np.array(boundaries['bottom'])[1:-1]
    F[...,1,1:-1] = F[...,1,1:-1] + np.array(boundaries['left'])[1:-1]
    F[...,-2,1:-1] = F[...,-2,1:-1] + np.array(boundaries['right'])[1:-1]
    
    return F[...,1:-1,1:-1].reshape(list(F[...,1:-1,1:-1].shape[:-2]) + [np.prod(F[...,1:-1,1:-1].shape[-2:])])
 
def generate_dataset(batch_size, n, h, boundaries, n_batches = 1, rhs_range = [-1,1]):
    lhs = tf.constant(poisson_matrix(n,n), dtype=tf.float64)
    lhs_chol = tf.linalg.cholesky(lhs)
    
    def chol(r):
        return tf.linalg.cholesky_solve(lhs_chol, tf.transpose(tf.stack([r])))
    
    @tf.contrib.eager.defun
    def chol_solve(rhs_arr):
        return tf.map_fn(chol, rhs)
    
    #pdb.set_trace()
    with ThreadPool(n_batches) as pool:
        F = pool.map(generate_random_RHS, zip(itertools.repeat(batch_size, n_batches), itertools.cycle(np.arange(5, 5 + n_batches//2)), itertools.repeat(n), itertools.repeat(5), itertools.repeat([0,n*h,0,n*h])))
        rhs = tf.concat(pool.map(poisson_RHS, zip(F, itertools.repeat(boundaries), itertools.repeat(h))), axis=0)
    print('RHSes generated.')
    
    soln = np.zeros((n_batches * batch_size, n, n), dtype = np.float64)
    soln[...,:,0] = boundaries['top']
    soln[...,:,-1] = boundaries['bottom']
    soln[...,0,:] = boundaries['left']
    soln[...,-1,:] = boundaries['right']
    soln[:,1:-1,1:-1] = tf.reshape(chol_solve(rhs), (n_batches * batch_size, n-2, n-2))
    #soln = chol_solve(rhs)
    print('solutions generated')
    
    #return tf.reshape(soln, (n_batches, batch_size, lhs_chol.shape[0])), tf.reshape(F, (n_batches, batch_size, F.shape[-2], F.shape[-1]))
    return tf.reshape(soln, (n_batches*batch_size, 1, soln.shape[-2], soln.shape[-1])), tf.reshape(tf.concat(F, axis = 0), (n_batches*batch_size, 1, F[0].shape[-2], F[0].shape[-1]))


#_, outputpath, ntest, h, batch_size, n_batches = sys.argv
parser = argparse.ArgumentParser(description = "Generate a series of Poisson equation RHS-solution pairs with specified Dirichlet boundary conditions on square domains")
parser.add_argument('-n', help = "No of gridpoints per side", required = True)
parser.add_argument('-h', help = "Grid spacing", required = True)
parser.add_argument('-t', help = "No of parallel processing threads ", required = False, default = 20)
parser.add_argument('-bs', '--batch_size' ,help = "Grid spacing", required = True)
args = parser.parse_args()

ntest = args.n
print(ntest)
h = float(h)
batch_size = int(batch_size)
n_batches = int(n_batches)

folder = 'dataset_' + str(ntest)
boundary_top = Boundary1D('Dirichlet', [(0,ntest*h),(ntest*h,ntest*h)], orientation='clockwise', RHS_function=lambda t: t-t, boundary_rhs_is_parametric=True)
boundary_right = Boundary1D('Dirichlet', [(ntest*h,ntest*h),(ntest*h,0)], orientation='clockwise', RHS_function=lambda t: t-t, boundary_rhs_is_parametric=True)
boundary_bottom = Boundary1D('Dirichlet', [(ntest*h,0),(0,0)], orientation='clockwise', RHS_function=lambda t: t-t, boundary_rhs_is_parametric=True)
boundary_left = Boundary1D('Dirichlet', [(0,0),(0,ntest*h)], orientation='clockwise', RHS_function=lambda t: t-t, boundary_rhs_is_parametric=True)

t0 = time.time()
soln,F = generate_dataset(batch_size=batch_size, n = ntest, h = h, n_batches=n_batches, boundaries={'top': boundary_top.RHS_evaluate(np.linspace(boundary_top.t.min(),boundary_top.t.max(),ntest)), 'right': boundary_right.RHS_evaluate(np.linspace(boundary_right.t.min(),boundary_right.t.max(),ntest)), 'bottom': boundary_bottom.RHS_evaluate(np.linspace(boundary_bottom.t.min(),boundary_bottom.t.max(),ntest)), 'left': boundary_left.RHS_evaluate(np.linspace(boundary_left.t.min(),boundary_left.t.max(),ntest))})
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
