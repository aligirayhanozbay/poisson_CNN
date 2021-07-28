import numpy as np
import tensorflow as tf
import itertools
import opt_einsum as oe
from multiprocessing import Pool as ThreadPool

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
        c = tf.constant(np.array([np.array(0.5*(domain[n+1] - domain[n]),dtype=np.float64) for n in range(0,len(domain),2)]), dtype = tf.keras.backend.floatx()) #scaling coefficients - for handling domains other than [-1,1] x [-1,1]
        d = tf.constant(np.array([np.array(0.5*(domain[n+1] + domain[n]),dtype=np.float64) for n in range(0,len(domain),2)]), dtype = tf.keras.backend.floatx())
        self.quadpts = tf.constant(np.apply_along_axis(lambda x: x + d, 0, oe.contract('i...,i->i...',tf.stack(np.meshgrid(*list(itertools.repeat(quadrature_x,ndims)),indexing = 'xy'), axis=0),c, backend = 'tensorflow')).transpose(list(np.arange(1,ndims+1)) + [0]),dtype = tf.keras.backend.floatx())
    
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
        coefficients.append(-integrator(integrand) * tf.cast(two_to_the_power_ndims,tf.keras.backend.floatx()) / (tf.cast(domain_volume,tf.keras.backend.floatx()) * tf.reduce_sum(tf.square(mplus1_pi_over_L[i]))))
    
    return tf.constant(np.array(coefficients), dtype = tf.keras.backend.floatx())

def homogeneous_analytical_dataset(rhs = 'random', output_shape = (64,64), nmodes = (16,16), domain = [1,1], n_threads = 1, return_rhs = True, max_magnitude = np.inf, batch_size = 1, expanded_dims = False, return_dx = False):
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
    -max_magnitude: If rhs is 'random', the solution will be scaled such that the maximum absolute value of the output will be this value.
    -expanded_dims: if set to True, the values returned will have an extra axis of length 1 as the axis #1 (not #0!)
    '''
    
    coords = [tf.linspace(tf.constant(0.0,dtype = tf.keras.backend.floatx()),domain[i], output_shape[i]) for i in range(len(output_shape))] #Evaluate coordinates along each axis
    coord_meshes = tf.stack(tf.meshgrid(*coords)) #Create meshgrid
    ndims = len(domain) #No of dims of function
    mode_permutations = tf.constant(np.array(list(itertools.product(*[np.arange(nmodes[i]) for i in range(len(nmodes))]))), dtype = tf.keras.backend.floatx()) #Every permutation of Fourier modes along different axes possible
    mplus1_pi_over_L = oe.contract('j,ij->ij',tf.constant(domain, dtype = tf.keras.backend.floatx())**(-1),(mode_permutations + 1)*np.pi, backend = 'tensorflow') #compute (m_i+1)*pi/L_i
    #print(mplus1_pi_over_L)
    #print(coord_meshes)
    sine_vals = tf.reduce_prod(tf.sin(oe.contract('ij,j...->ij...', mplus1_pi_over_L, coord_meshes, backend = 'tensorflow')),axis=1) #sin((m_1+1)*pi*x_1/L_1) * ... * sin((m_n+1)*pi*x_n/L_n))

    if rhs == 'random': #Generate random Fourier coefficients for a random RHS
        #For an n dimensional RHS function F(x_1,...x_n) = \sum_{m_1,...,m_n} (A_{m_1,...,m_n} * sin((m_1+1)*pi*x_1/L_1) * ... * sin((m_n+1)*pi*x_n/L_n))
        #\int_0^{L_n} ... \int_0^{L_1} F * sin((m_1+1)*pi*x_1/L_1) * ... * sin((m_n+1)*pi*x_n/L_n) dx_1 ... dx_n = A*L_1*...*L_n/2^n
        #Thus the solution to the Poisson equation \nabla^2 u = F will be u = a_{m_1,...,m_n} * sin((m_1+1)*pi*x_1/L_1) * ... * sin((m_n+1)*pi*x_n/L_n))
        #where a_{m_1,...,m_n} = -A_{m_1,...,m_n} * /(((m_1+1)*pi/L_1)^2+...+((m_n+1)*pi/L_n)^2)
        rhs_function_coeffs = oe.contract('ij,j->ij',2*tf.random.uniform(tf.stack([batch_size, mode_permutations.shape[0]]),dtype = tf.keras.backend.floatx())-1,tf.exp(-tf.reduce_sum(tf.cast(mode_permutations, tf.keras.backend.floatx()),axis=1)), backend = 'tensorflow') #Random RHS function Fourier coefficients
        soln_function_coeffs = -oe.contract('ij,j->ij',rhs_function_coeffs, tf.reduce_sum(tf.square(mplus1_pi_over_L),axis=1)**(-1), backend = 'tensorflow') #Random solution Fourier coefficients
        
        
        rhs = tf.Variable(oe.contract('ij,j...->i...', rhs_function_coeffs, sine_vals, backend = 'tensorflow'))
        soln = tf.Variable(oe.contract('ij,j...->i...', soln_function_coeffs, sine_vals, backend = 'tensorflow'))
        maxminratio = np.zeros((batch_size))
        
        if max_magnitude != np.inf:
            for i in range(int(rhs.shape[0])):
                scaling_factor = max_magnitude/tf.reduce_max(tf.abs(rhs[i,...]))
                maxminratio[i] = tf.abs(tf.reduce_max(rhs[i,...]) / tf.reduce_min(rhs[i,...]))
                rhs[i,...].assign(rhs[i,...] * scaling_factor)
                soln[i,...].assign(soln[i,...] * scaling_factor)

    elif callable(rhs):
        if np.prod(nmodes) % n_threads != 0:
            raise(ValueError('n_threads must divide reduce_prod(nmodes)'))
        
        pool = ThreadPool(n_threads) 
        full_domain = np.zeros((len(domain)*2)) #add lower bound 0s to domain argument for compatibility with integrate_nd
        full_domain[1::2] += np.array(domain)
        
        itg = integrator_nd(domain = full_domain)
        
        soln_function_coeffs = tf.concat(list(map(mode_coeff_calculation_multiprocessing_wrapper, zip(itertools.repeat(rhs,n_threads), itertools.repeat(tf.reduce_prod(domain),n_threads), tf.split(mplus1_pi_over_L, n_threads, axis=0), itertools.repeat(itg,n_threads), itertools.repeat(2**ndims,n_threads)))),axis=0)
        
        #soln_function_coeffs = tf.concat(pool.map(mode_coeff_calculation_multiprocessing_wrapper, zip(itertools.repeat(rhs,n_threads), itertools.repeat(tf.reduce_prod(domain),n_threads), tf.split(mplus1_pi_over_L, n_threads, axis=0), itertools.repeat(itg,n_threads), itertools.repeat(2**ndims,n_threads))),axis=0)

        soln = oe.contract('i,i...->...', soln_function_coeffs, sine_vals, backend = 'tensorflow')
        rhs = rhs(*coord_meshes)
    else:
        raise(TypeError('rhs must either be a callable function to be evaluated or random'))

    if expanded_dims:
        rhs = tf.expand_dims(rhs, axis = 1)
        soln = tf.expand_dims(soln, axis = 1)
    inp = []
    if return_rhs:
        inp.append(rhs)
    if return_dx:
        dx = [[domain[k]/(output_shape[k]-1) for k in range(len(domain))]]
        dx = tf.tile(tf.constant(dx, dtype = tf.keras.backend.floatx()), [batch_size, 1])
        inp.append(dx)
    return inp, soln
            
        
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

class homogeneous_analytical_dataset_generator(tf.keras.utils.Sequence):
    def __init__(self, batch_size = 1, batches_per_epoch = 1, nmodes = 'random', domain = 'random', output_shape = 'random', randomize_rhs_max_magnitude = False, nmodes_random_range = [[8,24],[8,24]], random_output_shape_range = [[64,85],[64,85]], domain_random_range = [[0.5,1.5],[0.5,1.5]], rhs_random_max_magnitude = 1.0, keep_dx_constant = True, **analytical_dataset_arguments):
        '''
        Creates a generator that can be used to generate infinitely many sets of Poisson eq. RHSes-BCs-solutions
        
        batch_size: Int. Batch size of outputs.
        batches_per_epoch: Int. No of batches to create in each training epoch.
        randomize_rhs_smoothness: Boolean. Set to True if it's desired to have random RHS smoothnesses.
        rhs_random_smoothness_range: List of 2 integers. First integer is the lower bound and 2nd integer is the upper bound of the random RHS smoothnesses. Ignored if randomize_rhs_smoothness is False.
        randomize_boundary_smoothness: Boolean. Set to True if it's desired to have random BC smoothnesses.
        boundary_random_smoothness_range: Dict containing entries 'left', 'top', 'right', 'bottom'. Each entry must be the same format as rhs_random_smoothness_range
        randomize_rhs_max_magnitude: Boolean. Set to True if it's desired to have random RHS magnitudes.
        rhs_random_max_magnitude: Float. Max value of the random max magnitude.
        randomize_boundary_max_magnitude: Boolean. Set to True if it's desired to have random BC magnitudes.
        boundary_random_max_magnitude: Dict containing entries 'left', 'top', 'right', 'bottom'. Each entry must be the same format as rhs_random_max_magnitude.
        numerical_dataset_arguments: Arguments to pass onto the function numerical_dataset.
        return_keras_style: If set to True, the values from the boundaries dict will be unpacked (in the order left-top-right-bottom) into members of the output list.
        exclude_zero_boundaries: If set to True, nonzero boundaries are not returned if return_keras_style is True.
        '''

        self.nmodes = nmodes
        if self.nmodes == 'random':
            self.nmodes_random_range = nmodes_random_range
        self.domain = domain
        if self.domain == 'random':
            self.domain_random_range = domain_random_range
        self.output_shape = output_shape
        if self.output_shape == 'random':
            self.random_output_shape_range = random_output_shape_range
        self.randomize_rhs_max_magnitude = randomize_rhs_max_magnitude
        if self.randomize_rhs_max_magnitude:
            self.rhs_random_max_magnitude = rhs_random_max_magnitude

        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.ada = analytical_dataset_arguments
        self.keep_dx_constant = keep_dx_constant

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx): #Generates a batch. Input idx is ignored but is necessary per Keras API.

        if self.randomize_rhs_max_magnitude:
            self.ada['rhs_max_magnitude'] = np.random.rand() * self.rhs_random_max_magnitude

        if self.nmodes == 'random':
            self.ada['nmodes'] = tuple([int(np.random.randint(self.nmodes_random_range[k][0], high = self.nmodes_random_range[k][1])) for k in range(len(self.nmodes_random_range))])
        else:
            self.ada['nmodes'] = self.nmodes

        if self.output_shape == 'random':
            self.ada['output_shape'] = tuple([int(np.random.randint(self.random_output_shape_range[k][0], high = self.random_output_shape_range[k][1])) for k in range(len(self.random_output_shape_range))])
        else:
            self.ada['output_shape'] = self.output_shape

        if self.domain == 'random' and self.keep_dx_constant:
            self.ada['domain'] = [np.random.rand()*(self.domain_random_range[0][1] - self.domain_random_range[0][0]) + self.domain_random_range[0][0]]
            dx = self.ada['domain'][0]/(self.ada['output_shape'][0] - 1)
            for k in range(1, len(self.domain_random_range)):
                self.ada['domain'].append(dx * (self.ada['output_shape'][k] - 1))
        elif self.domain == 'random':
            self.ada['domain'] = [np.random.rand()*(self.domain_random_range[k][1] - self.domain_random_range[k][0]) + self.domain_random_range[k][0] for k in range(len(self.domain_random_range))]
        
            
        inp, out = homogeneous_analytical_dataset(**self.ada, batch_size = self.batch_size)

        if self.keep_dx_constant:
            inp[1] = tf.expand_dims(inp[1][:,0], axis = 1)
        return inp, out
        

