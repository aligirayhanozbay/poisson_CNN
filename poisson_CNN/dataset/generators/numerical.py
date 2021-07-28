import tensorflow as tf
import itertools
import numpy as np
import multiprocessing
from collections.abc import Iterable

from ..solvers import *
from ..utils import *

def generate_random_RHS(batch_size, n_outputpts, smoothness = None, resize_method = tf.image.ResizeMethod.BICUBIC, max_magnitude = np.inf):
    
    '''
    This function generates random smooth RHS 'functions' defined pointwise by creating a random number field the shape of which is defined by n_controlpts and then super-sampling it using tf.image.resize_images. Results are stacked across the 1st dimension.
    
    batch_size: no. of random RHSes to generate
    n_outputpts: Iterable. No. of gridpoints in each direction of the output 
    smoothness: None or Iterable or int. No. of control pts of the spline along each dimension. Smaller values lead to 'smoother' results. If None, a value between 5 and n_outputpts[k]//1.5 is chosen randomly.
    resize_method Supersampling method for tf.image.resize_images. Use bicubic for smooth RHSes - bilinear or nearest neighbor NOT recommended
    max_random_magnitude: If the maximum absolute value of a generated random RHS exceeds this value, the entire thing is rescaled so the max abs. val. becomes this value
    '''
    n_controlpts = smoothness
    if n_controlpts == None:
        n_controlpts = [np.random.randint(5,int(n_outputpts[k]//1.5)) for k in range(len(n_outputpts))]
    elif isinstance(n_controlpts, int):
        try:
            n_controlpts = tf.ones(n_outputpts.shape, dtype = tf.int32)*n_controlpts
        except:
            n_controlpts = tf.ones(len(n_outputpts), dtype = tf.int32)*n_controlpts
    rhs = 2*tf.random.uniform([batch_size, 1] + list(n_controlpts), dtype = tf.keras.backend.floatx())-1
    rhs = image_resize(rhs, n_outputpts, resize_method = resize_method)
    rhs = rhs
    if max_magnitude != np.inf:
        rhs = set_max_magnitude_in_batch(rhs, max_magnitude)

    return rhs

def generate_random_boundaries(n_outputpts, batch_size = 1, max_magnitude = {'left':np.inf, 'top':np.inf, 'right':np.inf, 'bottom':np.inf}, smoothness = None, nonzero_boundaries = ['left', 'right', 'bottom', 'top'], return_with_expanded_dims = False, data_format = 'channels_first'):
    '''
    This function generates random smooth BC 'functions' defined pointwise by creating a random number field the size of which is determined by 'smoothness' and then super-sampling it using tf.image.resize_images. Results are stacked across the 1st dimension.
    
    batch_size: no. of random RHSes to generate
    n_outputpts: Iterable or int. No. of gridpoints in each direction of the output 
    resize_method: Supersampling method for tf.image.resize_images. Use bicubic for smooth boundaries - bilinear or nearest neighbor NOT recommended
    max_random_magnitude: Dict with entries 'left', 'right', 'top', 'bottom'. If the maximum absolute value of a generated random BC exceeds this value, it's rescaled so the max abs. val. becomes this value.
    nonzero_boundaries: Boundaries not included in this list will be all 0s. Entries same as max_random_magnitude.
    return_with_expanded_dims: If set to true, the returned arrays will have shape (batch_size,1,n) instead of (batch_size,n)
    data_format: If set to 'channels_first', the expanded dim will be the 2nd dimension. Else, it'll be the 3rd.
    
    Outputs a dict containing entries 'left', 'right', 'bottom' and 'top' with values as tf.Tensors of shape (batch_size, 1, n) or (batch_size, n) depending on whether return_with_expanded_dims is True, where n is the entry of n_outputpts corresponding to the boundary in question
    '''

    boundary_lengths = {'left' : n_outputpts[1], 'right' : n_outputpts[1], 'top' : n_outputpts[0], 'bottom' : n_outputpts[0]}
    if isinstance(smoothness, int):
        smoothness = {'left' : smoothness, 'right' : smoothness, 'top' : smoothness, 'bottom' : smoothness}
    elif smoothness == None:
        smoothness = {'left' : np.random.randint(5,int(n_outputpts[1]//1.5)), 'right' : np.random.randint(5,int(n_outputpts[1]//1.5)), 'top' : np.random.randint(5,int(n_outputpts[0]//1.5)), 'bottom' : np.random.randint(5,int(n_outputpts[0]//1.5))}
    boundaries = {}
    for boundary in boundary_lengths.keys():
        if boundary in nonzero_boundaries:
            boundaries[boundary] = image_resize(2*tf.random.uniform((batch_size,smoothness[boundary]), dtype = tf.keras.backend.floatx())-1, [batch_size, boundary_lengths[boundary]])

            if max_magnitude[boundary] != np.inf:
                boundaries[boundary] = set_max_magnitude_in_batch(boundaries[boundary], max_magnitude[boundary])

        else:
            boundaries[boundary] = tf.zeros((batch_size, boundary_lengths[boundary]), dtype = tf.keras.backend.floatx())
        if return_with_expanded_dims:
            if data_format == 'channels_first':
                boundaries[boundary] = tf.expand_dims(boundaries[boundary], axis = 1)
            else:
                boundaries[boundary] = tf.expand_dims(boundaries[boundary], axis = 2)
    return boundaries

def numerical_dataset(batch_size = 1, output_shape = 'random', dx = 'random', boundaries = 'random', rhses = 'random', rhs_smoothness = None, boundary_smoothness = None, rhs_max_magnitude = 1.0, boundary_max_magnitude = {'left':1.0, 'top':1.0, 'right':1.0, 'bottom': 1.0}, nonzero_boundaries = ['left', 'right', 'bottom', 'top'], solver_method = 'multigrid', return_rhs = True, return_boundaries = False, return_dx = False, return_shape = False, random_output_shape_range = [[64,85],[64,85]], random_dx_range = [0.005,0.05], normalize_by_domain_size = False, uniformly_distributed_aspect_ratios = True):
    '''
    Generates Poisson equation RHS-solution pairs with 'random' RHS functions.

    batch_size: No of problem-soln pairs to generate
    output_shape: shape of the outputs
    dx: Grid spacing. Can be 'random' for random grid spacings, a floating point value or an array of size (batch_size).
    boundaries: Supply the boundaries as a dict containing the entries 'top', 'bottom', 'right' and 'left' or set as 'random' for random BCs.
    rhses: Supply the RHSes as a tensor of shape (batch_size, 1, nx, ny) or set as 'random'
    smoothness: Int. Determines the resolution of the coarse grid.
    rhs_max_magnitude: Float. Max magnitude for generated rhses
    boundary_max_magnitude: Dict containing the entries 'top', 'bottom', 'right' and 'left'. Sets max magnitude for the corresponding boundaries.
    nonzero_boundaries: See generate_random_boundaries
    solver_method: 'multigrid' for PyAMG multigrid solver, 'multigrid_gpu' for pyamgx GPU multigrid solver, 'cholesky' for Cholesky decomposition solver (on GPU) or supply a callable taking arguments (rhses, boundaries, dx)
    return_rhs: If set to True, the RHSes will be appended to the output
    return_boundaries: If set to True, the BCs will be appended to the output
    return_dx: If set to True, the grid spacing(s) will be appended to the output
    random_output_shape_range: List of lists, where each list contains [n_min, n_max] along the corresponding dim
    normalize_by_domain_size: If set to True, the solution (and ONLY the solution) will be divided by the product of the domain lengths

    Outputs a tf.Tensor of shape (batch_size * smoothness_levels, n[0], n[1])
    '''
    
    if (isinstance(output_shape, str) and output_shape == 'random') and uniformly_distributed_aspect_ratios:
        aspect_ratios = generate_uniformly_distributed_aspect_ratios(random_output_shape_range, dx_range = None, samples = 1)
        output_shape, dx_generated = generate_output_shapes_and_grid_spacings_from_aspect_ratios(aspect_ratios, random_output_shape_range, [random_dx_range], constant_dx = True, samples = batch_size)
        output_shape = np.array(output_shape).tolist()
        dx = dx_generated[:,:1] if dx == 'random' else dx
    else:
        if isinstance(output_shape, str) and output_shape == 'random':
            output_shape = [np.random.randint(random_output_shape_range[k][0], high = random_output_shape_range[k][1]) for k in range(len(random_output_shape_range))]

        if isinstance(dx, str) and dx == 'random':
            dx = tf.random.uniform((batch_size,1))*(random_dx_range[1] - random_dx_range[0]) + random_dx_range[0]
        elif isinstance(dx, float):
            dx = np.ones((batch_size,1))*dx
            
    if isinstance(rhses,str) and rhses == 'random':
        rhses = generate_random_RHS(batch_size, output_shape, smoothness = rhs_smoothness, max_magnitude = rhs_max_magnitude)
    elif isinstance(rhses,str) and rhses == 'zero':
        rhses = np.zeros([batch_size,1] + list(output_shape))
    
    if boundaries == 'random':
        boundaries = generate_random_boundaries(output_shape, batch_size = batch_size, max_magnitude = boundary_max_magnitude, return_with_expanded_dims = True, nonzero_boundaries = nonzero_boundaries, smoothness = boundary_smoothness)
    elif boundaries == 'zero' or boundaries == 'homogeneous':
        boundaries = generate_random_boundaries(output_shape, batch_size = batch_size, return_with_expanded_dims = True, nonzero_boundaries = [])
    
    if not callable(solver_method):
        if solver_method == 'cholesky':
            solver_method = cholesky_poisson_solve
        elif solver_method == 'multigrid':
            solver_method = multigrid_poisson_solve
        elif solver_method == 'multigrid_gpu':
            solver_method = lambda *args: multigrid_poisson_solve(*args, use_pyamgx = True)
        else:
            raise(ValueError('solver_method must be a function or one of cholesky, multigrid or multigrid_gpu'))

    out = tf.constant(solver_method(rhses, boundaries,dx), dtype = tf.keras.backend.floatx())

    if normalize_by_domain_size:
        domainsize = tf.squeeze(dx**len(output_shape)) * np.prod(np.array(output_shape)-1)
        out = 10*tf.einsum('i...,i->i...', out, tf.cast(1/domainsize, out.dtype))
    
    inp = []
    if return_rhs:
        inp.append(tf.cast(rhses, tf.keras.backend.floatx()))
    if return_boundaries:
        inp.append(boundaries)
    if return_dx:
        inp.append(tf.cast(dx, tf.keras.backend.floatx()))
    if return_shape:
        inp.append(tf.cast(tf.constant(rhses.shape), tf.int32))
    
    if len(inp) > 0:
        return inp, out
    else:
        return out

class numerical_dataset_generator(tf.keras.utils.Sequence):
    def __init__(self, batch_size = 1, batches_per_epoch = 1, randomize_rhs_smoothness = False, rhs_random_smoothness_range = [5,20], randomize_boundary_smoothness = False, boundary_random_smoothness_range = {'left':[5,20], 'top':[5,20], 'right':[5,20], 'bottom': [5,20]}, randomize_rhs_max_magnitude = False, rhs_random_max_magnitude = 1.0, randomize_boundary_max_magnitudes = False, boundary_random_max_magnitudes = {'left':1.0, 'top':1.0, 'right':1.0, 'bottom': 1.0}, return_keras_style = True, exclude_zero_boundaries = False, **numerical_dataset_arguments):
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

        self.randomize_rhs_smoothness = randomize_rhs_smoothness
        if self.randomize_rhs_smoothness:
            self.rhs_random_smoothness_range = rhs_random_smoothness_range
        self.randomize_boundary_smoothness = randomize_boundary_smoothness
        if self.randomize_boundary_smoothness:
            self.boundary_random_smoothness_range = boundary_random_smoothness_range
        self.randomize_rhs_max_magnitude = randomize_rhs_max_magnitude
        if self.randomize_rhs_max_magnitude:
            self.rhs_random_max_magnitude = rhs_random_max_magnitude
        self.randomize_boundary_max_magnitudes = randomize_boundary_max_magnitudes
        if self.randomize_boundary_max_magnitudes:
            self.boundary_random_max_magnitudes = boundary_random_max_magnitudes

        self.exclude_zero_boundaries = exclude_zero_boundaries
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.nda = numerical_dataset_arguments
        self.return_keras_style = return_keras_style

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, idx): #Generates a batch. Input idx is ignored but is necessary per Keras API.
        if self.randomize_rhs_smoothness:
            self.nda['rhs_smoothness'] = np.random.randint(self.rhs_random_smoothness_range[0], high = self.rhs_random_smoothness_range[1])
        if self.randomize_boundary_smoothness:
            self.nda['boundary_smoothness'] = dict(zip(list(self.boundary_random_smoothness_range.keys()),[np.random.randint(brsr[0], high = brsr[1]) for brsr in list(self.boundary_random_smoothness_range.values())]))
        if self.randomize_rhs_max_magnitude:
            self.nda['rhs_max_magnitude'] = np.random.rand() * self.rhs_random_max_magnitude
        if self.randomize_boundary_max_magnitudes:
            self.nda['boundary_max_magnitude'] = dict(zip(list(self.boundary_random_max_magnitudes.keys()),[np.random.rand()*brmm for brmm in list(self.boundary_random_max_magnitudes.values())]))
        inp, out = numerical_dataset(**self.nda, batch_size = self.batch_size)
        if self.return_keras_style:
            if ('return_boundaries' in self.nda.keys()):
                if ('return_rhs' not in self.nda.keys()) or self.nda['return_rhs']:
                    boundary_location = 1
                else:
                    boundary_location = 0
                if (not self.exclude_zero_boundaries) or ('nonzero_boundaries' not in self.nda.keys()):
                    boundary_list = [inp[boundary_location]['left'], inp[boundary_location]['top'], inp[boundary_location]['right'], inp[boundary_location]['bottom']]
                else:
                    boundary_list = [inp[boundary_location][key] for key in self.nda['nonzero_boundaries']]
                _ = inp.pop(boundary_location)
                inp[boundary_location:boundary_location] = boundary_list
        return inp, out
