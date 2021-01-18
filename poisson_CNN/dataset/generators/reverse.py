import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import string

from ..utils import *
from ...utils import choose_conv_method

def handle_grid_parameters_range(value_range, ndims, output_dtype):
    value_range = tf.convert_to_tensor(value_range,dtype=output_dtype)
    if value_range.shape == 2:
        assert value_range[0] < value_range[1], 'Upper bound for grid spacings must be larger than or equal to the lower bound!'
        value_range = tf.expand_dims(value_range, 0)
        value_range = tf.tile(value_range, [ndims,1])
    assert value_range.shape[1] == 2, 'Dim 1 of grid_spacings range must be 2, containing the lower bound and upper bound for the respective spatial dimensions'
    assert value_range.shape[0] == ndims, '1st dim of value_range must have identical size to ndims'
    assert tf.reduce_all((value_range[:,1] - value_range[:,0]) >= 0), 'Dims ' + str(list(tf.reshape(tf.where((value_range[:,1]-value_range[:,0])<0), (-1,)).numpy())) + ' had upper bound of random range smaller than the lower bound'
    assert tf.reduce_all(value_range[:,0] >= 0), 'Dims ' + str(list(tf.reshape(tf.where(value_range[:,0]<0), (-1,)).numpy())) + ' had lower bounds of random range below 0'
    return value_range #shape of value_range is (ndims,2)

def process_normalizations(normalizations):

    normalization_types = ['rhs_max_magnitude', 'max_domain_size_squared', 'soln_max_magnitude']
    normalization_default_values = [False,False,False]

    if normalizations is None:
        return {key:default_val for key,default_val in zip(normalization_types,normalization_default_values)}
    elif isinstance(normalizations, dict):
        for key,default_val in zip(normalization_types,normalization_default_values):
            if key not in normalizations:
                normalizations[key] = default_val
        if isinstance(normalizations['rhs_max_magnitude'],bool) and normalizations['rhs_max_magnitude']:
            normalizations['rhs_max_magnitude'] = tf.constant(1.0)
    return normalizations
        

@tf.function
def generate_polynomial_and_second_derivative(roots,degree,npts,domain_size):
    '''
    Generates a 1d polynomial in the domain [0,domain_size] of degree (degree) sampled on npts equispaced points.

    Inputs:
    -roots: float tensor of shape (n_roots,). Values should be between 0 and -1 such that the form of the resulting polynomial is (x+roots[0])*(x+roots[1])*... n_roots should be greater than or equal to degree.
    -degree: int. Degree of the polynomial. Should be less than or equal to n_roots.
    -npts: int. No of sampling points.
    -domain_size: float, larger than 0. Total size of the domain.
    '''

    #Build grid
    dx = 1/(tf.cast(npts,domain_size.dtype)-1)
    coords = tf.linspace(tf.constant(0.0,dtype=tf.keras.backend.floatx())-dx,1.0+dx,npts+2)#extra 2 points needed for numerical stability
    
    #Evaluate polynomial and differentiate
    coords_expanded = tf.expand_dims(coords,1)
    roots = tf.expand_dims(roots[:degree],0)
    factors = coords_expanded + roots
    
    p = tf.reduce_prod(factors,1)
    dp = tf.gradients(p,coords)[0]/domain_size
    ddp = tf.gradients(dp,coords)[0]/domain_size
    
    #Autodiff occasionally generates nan values. Replace these with interpolated values.
    nan_values = tf.math.is_nan(ddp)
    if tf.reduce_any(nan_values):
        interpolated_ddp = 0.5*(ddp[:-2] + ddp[2:])
        nan_value_indices = tf.where(nan_values)
        interpolated_values_to_replace_nans = tf.gather_nd(interpolated_ddp, nan_value_indices-1)
        ddp = tf.tensor_scatter_nd_update(ddp, nan_value_indices, interpolated_values_to_replace_nans)
    
    return p[1:-1],ddp[1:-1]

@tf.function
def batch_generate_polynomial_and_second_derivative(roots,degrees,npts,domain_size):
    return tf.map_fn(lambda x: generate_polynomial_and_second_derivative(x[0],x[1],npts,domain_size), (roots, degrees), dtype=(tf.keras.backend.floatx(),tf.keras.backend.floatx()))

@tf.function
def polynomials_and_their_2nd_derivatives(npts, poly_deg, domain_sizes, batch_size = 1, homogeneous_bc = False):
    '''
    Creates a batch batch_size of polynomials with degree poly_deg sampled on npts equispaced points within domains [0,domain_sizes[k]] and also returns these polynomials' 2nd derivatives as computed by tensorflow autodifferentiation.

    -npts: int. No of grid points in the output.
    -poly_deg: int. Degree of the polynomial.
    -domain_sizes: tf.Tensor of shape (batch_size,). determines the domains in which the polynomials are generated.
    -batch_size: int. No of samples to generate.
    -homogeneous_bc: bool. If enabled, the generated polynomials will have value 0 at the start and end of the domain.
    '''
    npts = npts * tf.ones((batch_size,),dtype=tf.int32)
        
    poly_deg -= 1
    if homogeneous_bc:
        roots = -tf.random.uniform((batch_size,poly_deg,poly_deg-1),dtype=tf.keras.backend.floatx())
        roots = tf.concat([tf.zeros((batch_size,tf.shape(roots)[1],1),dtype=tf.keras.backend.floatx()),-tf.ones((batch_size,tf.shape(roots)[1],1),dtype=tf.keras.backend.floatx()),roots],-1)
    else:
        roots = -tf.random.uniform((batch_size,poly_deg,poly_deg+2),dtype=tf.keras.backend.floatx())
    #roots = tf.einsum('i,i...->i...',domain_sizes,roots)
    degrees = tf.tile(tf.expand_dims(tf.range(2,poly_deg+2),0),[batch_size,1])
    p,ddp = tf.map_fn(lambda x: batch_generate_polynomial_and_second_derivative(x[0],x[1],x[2],x[3]), (roots,degrees,npts,domain_sizes), dtype=(tf.keras.backend.floatx(),tf.keras.backend.floatx()))
    return p,ddp

class reverse_poisson_dataset_generator(tf.keras.utils.Sequence):
    def __init__(self, batch_size, batches_per_epoch, random_output_shape_range, fourier_coeff_grid_size_range, taylor_degree_range, grid_spacings_range = None, ndims = None, homogeneous_bc = False, return_rhses = True, return_boundaries = True, return_dx = True, normalizations = None, uniform_grid_spacing = False):
        '''
        Generates batches of random Poisson equation RHS-BC-solutions by first generating a solution and then using finite difference schemes to generate the RHS. Smooth results are ensured by using a Fourier series approach.

        Inputs:
        -batch_size: int. Number of samples to generate each time __getitem__ is called
        -batches_per_epoch: int. Determines the number of batches to generate per keras epoch. New, random data are generated each time __getitem__ is called, so this is mostly for compatibility purposes
        -random_output_shape_range: List of 2 ints, list of list of 2 ints, or np.ndarray/tf.Tensor of shape (ndims,2). Determines the range of random values from which the size of each spatial dimension will be picked for each batch (i.e. random value range for the shape of the  spatial dimensions)
        -fourier_coeff_grid_size_range: List of 2 ints, list of list of 2 ints, or np.ndarray/tf.Tensor of shape (ndims,2). Determines the range from which the number of Fourier coefficients per dimension will be drawn
        -taylor_coeff_grid_size_range: same as fourier_coeff_grid_size_range but for taylor series component
        -grid_spacings_range:  List of 2 floats, list of list of 2 floats, or np.ndarray/tf.Tensor of shape (ndims,2). Determines the range of values that the grid spacings can take for each dimension.
        -ndims: int. Number of spatial dimensions.
        -stencil_size: Odd int or list of odd ints. Determines the size of the FD stencil to be used for each dimension.
        -homogeneous_bc: bool. If set to true, solutions with homogeneous BCs will be returned only.
        -return_rhses: bool. If set to true, RHSes will be returned
        -return_boundaries: bool. If set to true, BCs will be returned.
        -return_dx: bool. If set to true, grid spacings will be returned.
        -normalizations: None or dict. Determines which normalizations to apply to the resulting datasets. None applies no normalization. If a dict, the keys are the types of normalization and the values determine the configuration of the normalization. Types available are
            *rhs_max_magnitude: bool, or a float value. RHSes and solution are scaled by float(rhs_max_magnitude)/max(|RHS|) if not set to False.
            *max_domain_size_squared: bool. If set to true, the solutions are scaled by 1/(max_domain_size^2)
        '''
        self.batch_size = batch_size
        if ndims is None:
            for k in [random_output_shape_range,fourier_coeff_grid_size_range]:
                try:
                    ndims = len(k)
                except:
                    pass
        self.ndims = ndims
        self.batches_per_epoch = batches_per_epoch
        self.homogeneous_bc = homogeneous_bc

        self.grid_spacings_range = handle_grid_parameters_range(grid_spacings_range, ndims, tf.keras.backend.floatx())
        self.random_output_shape_range = handle_grid_parameters_range(random_output_shape_range, ndims, tf.int32)
        self.fourier_coeff_grid_size_range = handle_grid_parameters_range(fourier_coeff_grid_size_range, ndims, tf.int32)

        self.taylor_degree_range = handle_grid_parameters_range(taylor_degree_range, ndims, tf.int32)
        self.taylor_einsum_str = 'B' + ',B'.join(list(string.ascii_lowercase[-self.ndims:])) + '->B' + string.ascii_lowercase[-self.ndims:]

        self.return_rhses = return_rhses

        self.return_boundaries = return_boundaries
        if self.return_boundaries:
            self._boundary_slices = []
            for k in range(self.ndims):
                sl0 = [Ellipsis] + [slice(0,None) for k in range(k)] + [0] + [slice(0,None) for k in range(ndims-k-1)]
                sl1 = [Ellipsis] + [slice(0,None) for k in range(k)] + [-1] + [slice(0,None) for k in range(ndims-k-1)]
                self._boundary_slices.append(sl0)
                self._boundary_slices.append(sl1)

        self.return_dx = return_dx
        self.normalizations = process_normalizations(normalizations)
        self.uniform_grid_spacing = uniform_grid_spacing
        

    def __len__(self):
        return self.batches_per_epoch

    @tf.function
    def generate_grid_spacings(self):
        #grid spacings shape: (batch_size, ndims)
        grid_spacings = tf.random.uniform((self.batch_size, self.ndims), dtype = tf.keras.backend.floatx())
        grid_spacings = tf.einsum('ij,j->ij', grid_spacings, self.grid_spacings_range[:,1] - self.grid_spacings_range[:,0]) + self.grid_spacings_range[:,0]
        return grid_spacings

    @tf.function
    def generate_grid_sizes(self, grid_size_range):
        #grid sizes shape: (ndims,)
        grid_size_range = tf.cast(grid_size_range, tf.keras.backend.floatx())
        grid_sizes = tf.random.uniform((self.ndims,), dtype = tf.keras.backend.floatx())
        grid_sizes = (grid_size_range[:,1] - grid_size_range[:,0]) * grid_sizes + grid_size_range[:,0] + 1
        grid_sizes = tf.cast(grid_sizes, tf.int32)
        return grid_sizes

    @tf.function
    def generate_grid_sizes_and_spacings_with_uniform_AR(self):
        aspect_ratios = generate_uniformly_distributed_aspect_ratios(output_shape_range = self.random_output_shape_range, dx_range = None if self.uniform_grid_spacing else self.grid_spacings_range, samples = self.batch_size)
        output_shape, grid_spacings = generate_output_shapes_and_grid_spacings_from_aspect_ratios(aspect_ratios, self.random_output_shape_range, self.grid_spacings_range, constant_dx = self.uniform_grid_spacing, samples = self.batch_size)
        return output_shape, grid_spacings
    
    @tf.function
    def generate_soln_fourier(self):
        tiles = [self.batch_size] + [1 for _ in range(self.ndims)]

        #determine how many Fourier modes should be in each sample in batch
        coeff_grid_size_range = tf.expand_dims(self.fourier_coeff_grid_size_range,0)
        coeff_grid_size_range = tf.tile(coeff_grid_size_range,tiles)
        n_coefficients = tf.map_fn(self.generate_grid_sizes,coeff_grid_size_range,dtype=tf.int32)

        #pick output grid size and grid spacings
        output_shape, grid_spacings = self.generate_grid_sizes_and_spacings_with_uniform_AR()
        output_shape = tf.tile(tf.expand_dims(output_shape,0), tiles[:2])

        #generate solutions (and get associated Fourier coefficients
        solns, coeffs = tf.map_fn(lambda x: generate_smooth_function(self.ndims,x[0],x[1],homogeneous_bc = self.homogeneous_bc, normalize = False, return_coefficients = True, coefficients_return_shape = tf.reduce_max(n_coefficients,0)),(output_shape,n_coefficients),dtype=(tf.keras.backend.floatx(),tf.keras.backend.floatx()))
        
        return tf.expand_dims(solns,1), coeffs, grid_spacings
        
    @tf.function(experimental_relax_shapes = True)
    def generate_rhses_fourier(self, coefficients, grid_size, grid_spacings):
        #adjust coefficients from generate_soln_fourier to get RHS coefficients
        domain_sizes = tf.einsum('ij,j->ij',grid_spacings,tf.cast(grid_size,tf.keras.backend.floatx()))
        if self.homogeneous_bc:#determine how many Fourier modes there are maximally
            coefficients_size = tf.shape(coefficients)[1:]
        else:
            coefficients_size = tf.shape(coefficients)[2:]
        wavenumbers = [tf.cast(tf.range(1,coefficients_size[k]+1),tf.keras.backend.floatx()) for k in range(self.ndims)]#generate (n+1)*pi values/wavenumbers for each coefficient
        coefficient_adjustment = tf.stack(tf.meshgrid(*wavenumbers,indexing='ij'),0)*tf.constant(math.pi,tf.keras.backend.floatx())#adjust each coefficient by factor of [(n0+1)^2*pi^2/L0^2 + (n1+1)^2*pi^2/L1^2 + ...]
        coefficient_adjustment = -tf.einsum('ij,j...->i...',(1/domain_sizes)**2,coefficient_adjustment**2)
        if not self.homogeneous_bc:
            coefficient_adjustment = tf.expand_dims(coefficient_adjustment,1)
        coefficients = coefficients * coefficient_adjustment

        grid_size = tf.expand_dims(grid_size,0)
        grid_size = tf.tile(grid_size,[self.batch_size,1])

        #get the RHS given the adjusted coefficients
        rhs = tf.map_fn(lambda x: generate_smooth_function(self.ndims,x[0],x[1],homogeneous_bc = self.homogeneous_bc, normalize = False, return_coefficients = False), (grid_size, coefficients), dtype = tf.keras.backend.floatx())

        rhs = tf.expand_dims(rhs,1)

        return rhs, domain_sizes

    @tf.function(experimental_relax_shapes = True)
    def build_taylor_rhs_component(self, vals, indices):
        vals = [vals[k][indices[k]] for k in range(self.ndims)]
        return tf.einsum(self.taylor_einsum_str,*vals)
        
    @tf.function(experimental_relax_shapes = True)
    def generate_soln_and_rhs_taylor(self, grid_size, domain_sizes):
        #generate polynomials and their 2nd derivatives along each direction
        polynomials = []
        second_derivatives = []

        polynomial_degrees = self.generate_grid_sizes(self.taylor_degree_range)
        
        for k in range(self.ndims):
            coeffs = 2*tf.random.uniform([self.batch_size, polynomial_degrees[k]-1],dtype = tf.keras.backend.floatx())-1
            p, ddp = polynomials_and_their_2nd_derivatives(grid_size[k], polynomial_degrees[k], domain_sizes[:,k], batch_size = self.batch_size, homogeneous_bc = self.homogeneous_bc)
            p = tf.einsum('bij,bi->bj',p,coeffs)
            ddp = tf.einsum('bij,bi->bj',ddp,coeffs)
            polynomials.append(p)
            ddp = tf.stack([p,ddp],0)
            second_derivatives.append(ddp)

        #build soln = X(x)Y(y)Z(z) etc
        soln = tf.expand_dims(tf.einsum(self.taylor_einsum_str,*polynomials),1)

        #build rhs = X"YZ + XY"Z + XYZ" etc
        I = tf.eye(self.ndims,dtype = tf.int32)
        rhs = tf.map_fn(lambda x: self.build_taylor_rhs_component(vals = second_derivatives, indices = x), I, dtype = tf.keras.backend.floatx())
        rhs = tf.reduce_sum(rhs,0)
        rhs = tf.expand_dims(rhs,1)
        
        return rhs, soln
    
    #@tf.function
    def pack_outputs(self, rhses, solns, grid_spacings):
        #create output tuples according to configuration
        problem_definition = []
        if self.return_rhses:
            problem_definition.append(rhses)

        if self.return_boundaries:
            for sl in self._boundary_slices:
                problem_definition.append(solns[sl])

        if self.return_dx:
            if self.uniform_grid_spacing:
                grid_spacings = grid_spacings[...,:1]
            problem_definition.append(grid_spacings)

        return problem_definition, solns

    #@tf.function
    def rhs_max_magnitude_normalization(self,rhses,solns):
        rhses, scaling_factors = set_max_magnitude_in_batch_and_return_scaling_factors(rhses,self.normalizations['rhs_max_magnitude'])
        return rhses, scaling_factors

    #@tf.function
    def max_domain_size_squared_normalization(self,domain_sizes):
        scaling_factors = 1/tf.reduce_max(domain_sizes,1)**2
        return scaling_factors

    #@tf.function
    def apply_normalization(self,rhses,solns,domain_sizes):
        if (self.normalizations['rhs_max_magnitude'] != False):
            rhses, rhs_max_magnitude_soln_scaling_factors = self.rhs_max_magnitude_normalization(rhses,solns)
            solns = tf.einsum('i...,i->i...',solns,rhs_max_magnitude_soln_scaling_factors)
        if (self.normalizations['soln_max_magnitude'] != False):
            solns = set_max_magnitude_in_batch(solns,1.0)
        if self.normalizations['max_domain_size_squared']:
            max_domain_size_squared_soln_scaling_factors = self.max_domain_size_squared_normalization(domain_sizes)
            solns = tf.einsum('i...,i->i...',solns,max_domain_size_squared_soln_scaling_factors)
        return rhses, solns

    @tf.function(experimental_relax_shapes=True)
    def set_taylor_result_peak_magnitude_to_fourier_peak_magnitude(self, rhses_fourier, rhses_taylor, solns_taylor):
        rhses_taylor_max = tf.map_fn(lambda x: tf.reduce_max(tf.abs(x)),rhses_taylor)#scale taylor series component to that the max magnitude is identical to the fourier series component
        rhses_fourier_max = tf.map_fn(lambda x: tf.reduce_max(tf.abs(x)),rhses_fourier)
        taylor_scaling_coeffs = rhses_fourier_max/rhses_taylor_max
        taylor_scaling_coeffs = tf.reshape(taylor_scaling_coeffs, [self.batch_size] + [1 for _ in range(self.ndims+1)])
        rhses_taylor_out = taylor_scaling_coeffs * rhses_taylor
        solns_taylor_out = taylor_scaling_coeffs * solns_taylor
        return rhses_taylor_out, solns_taylor_out
        

    #@tf.function
    def __getitem__(self, idx = 0):
        #fourier component
        solns_fourier, soln_coeffs_fourier, grid_spacings_fourier = self.generate_soln_fourier()
        rhses_fourier, domain_sizes = self.generate_rhses_fourier(soln_coeffs_fourier, tf.shape(solns_fourier)[2:], grid_spacings_fourier)
        
        #taylor series component
        rhses_taylor, solns_taylor = self.generate_soln_and_rhs_taylor(tf.shape(solns_fourier)[2:], domain_sizes)
        rhses_taylor, solns_taylor = self.set_taylor_result_peak_magnitude_to_fourier_peak_magnitude(rhses_fourier, rhses_taylor, solns_taylor)
        
        #sum components
        solns = solns_fourier + solns_taylor
        rhses = rhses_fourier + rhses_taylor
        grid_spacings = grid_spacings_fourier
        
        #apply normalization
        rhses, solns = self.apply_normalization(rhses, solns, domain_sizes)
        
        #pack
        out = self.pack_outputs(rhses, solns, grid_spacings)
        
        return out

if __name__=='__main__':
    nmax = 300
    nmin = 150
    dmax = 0.1
    dmin = 0.01
    dxrange = [(1e+0)*dmin,(1e+0)*dmax]
    ndims = 2
    grid_size_range = [[nmin,nmax] for _ in range(ndims)]
    cmax = 10
    cmin = 5
    ctrl_pt_range = [[cmin,cmax] for _ in range(ndims)]
    hbc = True
    normalizations = {'rhs_max_magnitude':True}
    uniform_grid_spacing = True
    rpdg = reverse_poisson_dataset_generator(batch_size = 5, batches_per_epoch = 5, random_output_shape_range = grid_size_range, fourier_coeff_grid_size_range = ctrl_pt_range, taylor_degree_range = ctrl_pt_range, grid_spacings_range = dxrange, ndims = 2, homogeneous_bc = True, return_rhses = True, return_boundaries = False, return_dx = True, normalizations = normalizations, uniform_grid_spacing = uniform_grid_spacing)
    inp, out = rpdg.__getitem__()


    if uniform_grid_spacing:
        inp[1] = tf.concat([inp[1] for _ in range(ndims)],1)
    from ...losses import linear_operator_loss
    loss_fn = linear_operator_loss(stencil_sizes = 5, orders = 2, ndims = 2, data_format = 'channels_first')
    loss_val = loss_fn(inp[0],out,inp[1])
    print(loss_val)
    import pdb
    pdb.set_trace()

