import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import itertools, h5py
import opt_einsum as oe
from multiprocessing import Pool as ThreadPool
opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
conf = tf.ConfigProto(gpu_options=opts)
tf.enable_eager_execution(config=conf)

class integrator_nd:
    '''
    Class that integrates a given function f in an n-dimensional box.
    
    --Init arguments--
    -domain       : Defines the boundaries of the box. Format is [dim1 start, dim1 end, dim2 start, dim2 end, ..., dim_n start, dim_n end]
    -n_quadpts    : No of Gauss Legendre quadrature points across across each dimension
    
    --Call arguments--
    -f            : Callable taking n double-precision tensorflow tensors (shape of each: n_quadpts x n_quadpts x ... x n_quadpts, repeated n times) as arguments (n = dimensionality of the integration domain, i.e. len(domain)//2) and returning one double-precision tensorflow tensor of shape n_quadpts x n_quadpts x ... x n_quadpts, repeated n times. Represents the function to integrate. The inputs will be meshgrids containing the coordinates of the function to integrate, and output must be the values of the function.
    
    --Example--
    We want to integrate e^(-x^2-y^2-z^2) in the domain x = [1,2], y = [1,2], z = [1,2].
    First we need to define our function: g = lambda x,y,z: tf.exp(-x**2-y**2-z**2)
    Then we create the integreator: itg = integrator_nd(domain=[1.0,2.0,1.0,2.0,1.0,2.0])
    Finally, we evaluate the integral by calling the integrator: itg(g)
    '''
    
    def __init__(self, domain = [0,1,0,1], n_quadpts = 20):
        ndims = len(domain)//2
        quadrature_x, quadrature_w = tuple([np.polynomial.legendre.leggauss(n_quadpts)[i].astype(np.float64) for i in range(2)]) #quadrature weights and points
        c = tf.constant(np.array([np.array(0.5*(domain[n+1] - domain[n]),dtype=np.float64) for n in range(0,len(domain),2)]), dtype = tf.float64) #scaling coefficients - for handling domains other than [-1,1] x [-1,1]
        d = tf.constant(np.array([np.array(0.5*(domain[n+1] + domain[n]),dtype=np.float64) for n in range(0,len(domain),2)]), dtype = tf.float64)
        self.quadpts = tf.constant(np.apply_along_axis(lambda x: x + d, 0, oe.contract('i...,i->i...',tf.stack(np.meshgrid(*list(itertools.repeat(quadrature_x,ndims)),indexing = 'xy'), axis=0),c, backend = 'tensorflow')).transpose(list(np.arange(1,ndims+1)) + [0]),dtype = tf.float64)
    
        self.quadweights = tf.reduce_prod(c)*quadrature_w
    
        for i in range(1,ndims):
            self.quadweights = oe.contract('...,i->...i',self.quadweights,quadrature_w)
                                      
    def __call__(self, f):
        fi = f(*tf.unstack(self.quadpts, axis = -1))
        return tf.reduce_sum(tf.multiply(self.quadweights,fi))

def mode_coeff_calculation_multiprocessing_wrapper(args):
    #Integrates F(x1,x2,...,xn) * sin((m_1+1)*pi*x_1/L_1) * ... * sin((m_n+1)*pi*x_n/L_n)) dx_1 ... dx_n. Written as a separate function to permit multiprocessing pool map
    
    F = args[0]
    domain_volume = args[1]
    mplus1_pi_over_L = args[2]
    integrator = args[3]
    two_to_the_power_ndims = args[4]
    
    coefficients = []
    for i in range(int(mplus1_pi_over_L.shape[0])):
        integrand = lambda *vars: F(*vars) * tf.reduce_prod(tf.sin(oe.contract('i...,i->i...',tf.stack(vars),mplus1_pi_over_L[i],backend = 'tensorflow')),axis=0)
        coefficients.append(-integrator(integrand) * tf.cast(two_to_the_power_ndims,tf.float64) / (tf.cast(domain_volume,tf.float64) * tf.reduce_sum(tf.square(mplus1_pi_over_L[i]))))
    
    return tf.constant(np.array(coefficients), dtype = tf.float64)

def generate_analytical_solution_homogeneous_bc(rhs = 'random', output_shape = (64,64), nmodes = (16,16), domain = [1,1], n_threads = 16, rhs_return = True, max_random_magnitude = np.inf, n_random = 1):
    '''
    Generates an analytical solution to the Poisson equation with homogeneous Dirichlet (i.e. 0) Boundary conditions in the box x_1 = [0, L_1], ..., x_n = [0, L_n]
    Solution strategy is Fourier series based as outlined in
    https://en.wikiversity.org/wiki/Partial_differential_equations/Poisson_Equation#Solution_to_Case_with_4_Homogeneous_Boundary_Conditions
    
    --Inputs--
    -rhs: Callable compatible with integrator_nd, or 'random'. If a callable is given, an analytical  solution for that function will be generated. If 'random', a random function containing random Fourier modes defined in nmodes will be returned.
    -output_shape: Shape of the output
    -nmodes: # of Fourier modes to include for each direction
    -domain: [L_1,L_2,...L_n]
    -n_threads: no of multiprocessing threads. useful only in non-random rhses
    -rhs_return: If set to true, it'll also return the RHS source term of the Poisson equation sampled at the same points as the solution
    -max_random_magnitude: If rhs is 'random', the solution will be scaled such that the maximum absolute value of the output will be this value.
    '''
    
    
    #import inspect
    #frame = inspect.currentframe()
    #args, _, _, values = inspect.getargvalues(frame)
    #print('-----------------------')
    #print('function name ' + str(inspect.getframeinfo(frame)[2]))
    #for i in args:
    #    print(i + ' = ' + str(values[i]))
    
    coords = [tf.linspace(tf.constant(0.0,dtype = tf.float64),domain[i], output_shape[i]) for i in range(len(output_shape))] #Evaluate coordinates along each axis
    coord_meshes = tf.stack(tf.meshgrid(*coords)) #Create meshgrid
    ndims = len(domain) #No of dims of function
    mode_permutations = tf.constant(np.array(list(itertools.product(*[np.arange(nmodes[i]) for i in range(len(nmodes))]))), dtype = tf.float64) #Every permutation of Fourier modes along different axes possible
    mplus1_pi_over_L = oe.contract('j,ij->ij',tf.constant(domain, dtype = tf.float64)**(-1),(mode_permutations + 1)*np.pi, backend = 'tensorflow') #compute (m_i+1)*pi/L_i
    #print(mplus1_pi_over_L)
    #print(coord_meshes)
    sine_vals = tf.reduce_prod(tf.sin(oe.contract('ij,j...->ij...', mplus1_pi_over_L, coord_meshes, backend = 'tensorflow')),axis=1) #sin((m_1+1)*pi*x_1/L_1) * ... * sin((m_n+1)*pi*x_n/L_n))

    if rhs == 'random': #Generate random Fourier coefficients for a random RHS
        #For an n dimensional RHS function F(x_1,...x_n) = \sum_{m_1,...,m_n} (A_{m_1,...,m_n} * sin((m_1+1)*pi*x_1/L_1) * ... * sin((m_n+1)*pi*x_n/L_n))
        #\int_0^{L_n} ... \int_0^{L_1} F * sin((m_1+1)*pi*x_1/L_1) * ... * sin((m_n+1)*pi*x_n/L_n) dx_1 ... dx_n = A*L_1*...*L_n/2^n
        #Thus the solution to the Poisson equation \nabla^2 u = F will be u = a_{m_1,...,m_n} * sin((m_1+1)*pi*x_1/L_1) * ... * sin((m_n+1)*pi*x_n/L_n))
        #where a_{m_1,...,m_n} = -A_{m_1,...,m_n} * /(((m_1+1)*pi/L_1)^2+...+((m_n+1)*pi/L_n)^2)
        rhs_function_coeffs = oe.contract('ij,j->ij',2*tf.random.uniform(tf.stack([n_random, mode_permutations.shape[0]]),dtype = tf.float64)-1,tf.exp(-tf.reduce_sum(tf.cast(mode_permutations, tf.float64),axis=1)), backend = 'tensorflow') #Random RHS function Fourier coefficients
        soln_function_coeffs = -oe.contract('ij,j->ij',rhs_function_coeffs, tf.reduce_sum(tf.square(mplus1_pi_over_L),axis=1)**(-1), backend = 'tensorflow') #Random solution Fourier coefficients
        
        
        rhs = tf.Variable(oe.contract('ij,j...->i...', rhs_function_coeffs, sine_vals, backend = 'tensorflow'))
        soln = tf.Variable(oe.contract('ij,j...->i...', soln_function_coeffs, sine_vals, backend = 'tensorflow'))
        maxminratio = np.zeros((n_random))
        
        if max_random_magnitude != np.inf:
            for i in range(int(rhs.shape[0])):
                scaling_factor = max_random_magnitude/tf.reduce_max(tf.abs(rhs[i,...]))
                maxminratio[i] = tf.abs(tf.reduce_max(rhs[i,...]) / tf.reduce_min(rhs[i,...]))
                rhs[i,...].assign(rhs[i,...] * scaling_factor)
                soln[i,...].assign(soln[i,...] * scaling_factor)
        
        #print('---Peak positive magnitude to peak negative magnitude ratio---')
        #print(np.median(maxminratio[np.abs(maxminratio) < 1e4]))
            
        if rhs_return:
            return rhs, soln
        else:
            return soln
        
    elif callable(rhs):
        if np.prod(nmodes) % n_threads != 0:
            raise(ValueError('n_threads must divide reduce_prod(nmodes)'))
        
        pool = ThreadPool(n_threads) 
        full_domain = np.zeros((len(domain)*2)) #add lower bound 0s to domain argument for compatibility with integrate_nd
        full_domain[1::2] += np.array(domain)
        
        itg = integrator_nd(domain = full_domain)
        
        soln_function_coeffs = tf.concat(list(map(mode_coeff_calculation_multiprocessing_wrapper, zip(itertools.repeat(rhs,n_threads), itertools.repeat(tf.reduce_prod(domain),n_threads), tf.split(mplus1_pi_over_L, n_threads, axis=0), itertools.repeat(itg,n_threads), itertools.repeat(2**ndims,n_threads)))),axis=0)
        
        #soln_function_coeffs = tf.concat(pool.map(mode_coeff_calculation_multiprocessing_wrapper, zip(itertools.repeat(rhs,n_threads), itertools.repeat(tf.reduce_prod(domain),n_threads), tf.split(mplus1_pi_over_L, n_threads, axis=0), itertools.repeat(itg,n_threads), itertools.repeat(2**ndims,n_threads))),axis=0)

        
        if rhs_return:
            return rhs(*coord_meshes), oe.contract('i,i...->...', soln_function_coeffs, sine_vals, backend = 'tensorflow')
        else:
            return oe.contract('i,i...->...', soln_function_coeffs, sine_vals, backend = 'tensorflow')
    else:
        raise(TypeError('rhs must either be a callable function to be evaluated or random'))
        
def random_calculation_multiprocessing_wrapper(args):
    

    
    batch_size = args[0]
    solution_generator_parameters = args[1]
    rhses = []
    solns = []
    for i in range(batch_size):
        rhs, soln = generate_analytical_solution_homogeneous_bc(**solution_generator_parameters)
        rhses.append(rhs)
        solns.append(solns)
    return np.concatenate(rhses), np.concatenate(solns)
        
if __name__ == '__main__':
    import argparse
    import time
    parser = argparse.ArgumentParser(description = "Generate a series of analytical Poisson equation RHS-solution pairs with 0 Dirichlet boundary conditions on square domains")
    parser.add_argument('-o', help = "Path to output file", required = True)
    parser.add_argument('-n', help = "No of gridpoints per side, specified by a series of integers separated by spaces (e.g. -n 64 64 64)", required = True)
    parser.add_argument('-nm',help = "# of Fourier modes to include for each direction, specified by a series of integers separated by spaces (e.g. -n 16 16 16)")
    parser.add_argument('-dx', help = "Grid spacing. Must be supplied if --domain isn't.", required = False)
    parser.add_argument('-d', '--domain', help = "Domain extent, specified by a series of numbers separated by spaces (e.g. -d 1 2.2 0.5)", required = False)
    #parser.add_argument('-t', help = "No of parallel processing threads ", required = False, default = 1)
    parser.add_argument('-bs', '--batch_size' ,help = "No of solutions to generate", required = False, default = 1)
    parser.add_argument('-m', '--max_magnitude', help = "Max magnitude of generated solutions", required = False, default = np.inf)
    args = parser.parse_args()
    
    try:
        domain = [float(p) for p in args.domain.split(' ')]
        output_shape = [int(p) for p in args.n.split(' ')]
    except:
        dx = float(args.dx)
        output_shape = [int(p) for p in args.n.split(' ')]
        domain = [dx*(p-1) for p in output_shape]
        
    nmodes = [int(p) for p in args.nm.split(' ')]
    #n_threads = int(args.t)
    batch_size = int(args.batch_size)
    outputpath = args.o
    max_magnitude = np.float(args.max_magnitude)
    
    params = {'rhs' : 'random', 'output_shape' : output_shape, 'nmodes' : nmodes, 'domain' : domain, 'rhs_return' : True, 'max_random_magnitude' : max_magnitude, 'n_random' : batch_size}
    
    #pool = ThreadPool(n_threads)
    t0 = time.time()
    #res = pool.map(random_calculation_multiprocessing_wrapper, zip(itertools.repeat(batch_size, n_threads), itertools.repeat(params, n_threads)))
    #res = list(map(random_calculation_multiprocessing_wrapper, zip(itertools.repeat(batch_size, n_threads), itertools.repeat(params, n_threads))))
    #rhses, solns = zip(*list(map(random_calculation_multiprocessing_wrapper, itertools.repeat([batch_size, params], n_threads))))
    rhses, solns = generate_analytical_solution_homogeneous_bc(**params)
    t1 = time.time()
    print('Generation of training data took ' + str(t1-t0) + ' seconds')
    #pool.close()
    
    rhses = tf.expand_dims(tf.stack(rhses), axis = 1)
    solns = tf.expand_dims(tf.stack(solns), axis = 1)
    
    from conv_laplacian_loss import conv_laplacian_loss
    tf.keras.backend.set_floatx('float64')
    try:
        cll = conv_laplacian_loss(output_shape, dx)
    except:
        cll = conv_laplacian_loss(output_shape, domain[0]/(output_shape[0]-1))
    
    print('---conv laplacian loss---')
    print(cll(rhses, solns))
    with h5py.File(outputpath, 'w') as hf:
        hf.create_dataset('soln', data=solns)
        hf.create_dataset('F', data=rhses)
    print('Data saved.')
    print('Max RHS  : ' + str(tf.reduce_max(rhses)))
    print('Min RHS  : ' + str(tf.reduce_min(rhses)))
    print('Max soln : ' + str(tf.reduce_max(solns)))
    print('Min soln : ' + str(tf.reduce_min(solns)))