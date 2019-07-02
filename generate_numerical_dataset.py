import tensorflow as tf
import itertools
import numpy as np
import multiprocessing
from multigrid_poisson_solve import multigrid_poisson_solve
from cholesky_poisson_solve import cholesky_poisson_solve
from collections.abc import Iterable

@tf.function
def image_resize(image, newshape, data_format = 'channels_first', resize_method = tf.image.ResizeMethod.BICUBIC):
    imagedims = len(image.shape)
    if data_format == 'channels_first':
        if len(image.shape) == 4:
            image = tf.transpose(image, (0,2,3,1))
        elif len(image.shape) == 3:
            image = tf.expand_dims(image, axis = 3)
        elif len(image.shape) == 2:
            image = tf.expand_dims(tf.expand_dims(image, axis = 2), axis = 0)
    if isinstance(newshape, list) or isinstance(newshape, tuple) or len(newshape.shape) == 1:
        newshape = tf.tile(tf.constant([newshape]), tf.constant([image.shape[0],1]))
        
    out = tf.cast(tf.map_fn(lambda x: tf.compat.v1.image.resize_images(x[0],x[1],method=resize_method,align_corners=True), (image, newshape), parallel_iterations=multiprocessing.cpu_count(), dtype = tf.float32), image.dtype)
    
    if data_format == 'channels_first':
        if imagedims == 4:
            out = tf.transpose(out, (0,3,1,2))
        elif imagedims == 3:
            out = out[...,0]
        else:
            out = out[0,...,0]
            
    return out

@tf.function
def set_max_magnitude(arr, max_random_magnitude = None):
    if max_random_magnitude == None:
        max_random_magnitude = arr[1]
        arr = arr[0]
        
    scaling_factor = max_random_magnitude/tf.reduce_max(tf.abs(arr))
    return arr * scaling_factor
    
@tf.function
def set_max_magnitude_in_batch(arr, max_random_magnitude):
    if isinstance(max_random_magnitude, float):
        max_random_magnitude = tf.cast(tf.constant([max_random_magnitude for k in range(arr.shape[0])]), arr.dtype)
    
    return tf.map_fn(set_max_magnitude, (arr, max_random_magnitude), dtype = arr.dtype, parallel_iterations = multiprocessing.cpu_count())

def generate_random_RHS(batch_size, n_controlpts, n_outputpts, resize_method = tf.image.ResizeMethod.BICUBIC, max_random_magnitude = np.inf):
    
    '''
    This function generates random smooth RHS 'functions' defined pointwise by creating a random number field defined in n_controlpts and then super-sampling it using tf.image.resize_images. Results are stacked across the 1st dimension.
    
    batch_size: no. of random RHSes to generate
    n_controlpts: Iterable or int. No. of control pts of the spline along each dimension. Smaller values lead to 'smoother' results.
    n_outputpts: Iterable or int. No. of gridpoints in each direction of the output 
    supersample_method: Supersampling method for tf.image.resize_images. Use bicubic for smooth RHSes - bilinear or nearest neighbor NOT recommended
    max_random_magnitude: If the maximum absolute value of a generated random RHS exceeds this value, the entire thing is rescaled so the max abs. val. becomes this value
    '''
    
    try:
        n_controlpts = tf.ones(n_outputpts.shape, dtype = tf.int32)*n_controlpts
    except:
        n_controlpts = tf.ones(len(n_outputpts), dtype = tf.int32)*n_controlpts
    rhs = 2*tf.random.uniform([batch_size, 1] + list(n_controlpts), dtype = tf.keras.backend.floatx())-1
    print('small grid generated')
    rhs = image_resize(rhs, n_outputpts, resize_method = resize_method)
    print('resized')
    rhs = rhs
    print('RHS generated')
    if max_random_magnitude != np.inf:
        rhs = set_max_magnitude_in_batch(rhs, max_random_magnitude)
        print('Max magnitudes set')
    # if max_random_magnitude != np.inf:
    #     for i in range(int(rhs.shape[0])):
    #         scaling_factor = max_random_magnitude/tf.reduce_max(tf.abs(rhs[i,...]))
    #         rhs[i,...].assign(rhs[i,...] * scaling_factor)

    return rhs

def generate_random_boundaries(n_outputpts, batch_size = 1, max_random_magnitude = {'left':1.0, 'top':1.0, 'right':1.0, 'bottom': 1.0}, smoothness = None, nonzero_boundaries = ['left', 'right', 'bottom', 'top'], return_with_expanded_dims = False, data_format = 'channels_first'):
    boundary_lengths = {'left' : n_outputpts[1], 'right' : n_outputpts[1], 'top' : n_outputpts[0], 'bottom' : n_outputpts[0]}
    if isinstance(smoothness, int):
        smoothness = {'left' : smoothness, 'right' : smoothness, 'top' : smoothness, 'bottom' : smoothness}
    elif smoothness == None:
        smoothness = {'left' : np.random.randint(5,int(n_outputpts[1]//1.5)), 'right' : np.random.randint(5,int(n_outputpts[1]//1.5)), 'top' : np.random.randint(5,int(n_outputpts[0]//1.5)), 'bottom' : np.random.randint(5,int(n_outputpts[0]//1.5))}
    boundaries = {}
    for boundary in smoothness.keys():
        if boundary in nonzero_boundaries:
            boundaries[boundary] = tf.Variable(image_resize(2*tf.random.uniform((batch_size,smoothness[boundary]), dtype = tf.keras.backend.floatx())-1, [batch_size, boundary_lengths[boundary]]))
            # try:
            #     boundaries[boundary] = tf.Variable(tf.cast(tf.transpose(tf.squeeze(tf.image.resize_images(2*tf.random.uniform((smoothness[boundary],1,batch_size))-1,[boundary_lengths[boundary],1], method = tf.image.ResizeMethod.BICUBIC), axis = 1)), tf.keras.backend.floatx()))
            # except:
            #     boundaries[boundary] = tf.Variable(tf.cast(tf.transpose(tf.squeeze(tf.compat.v1.image.resize_images(2*tf.random.uniform((smoothness[boundary],1,batch_size))-1,[boundary_lengths[boundary],1], method = tf.image.ResizeMethod.BICUBIC), axis = 1)), tf.keras.backend.floatx()))
            if max_random_magnitude[boundary] != np.inf:
                boundaries[boundary] = set_max_magnitude_in_batch(boundaries[boundary], max_random_magnitude[boundary])
                # for i in range(int(boundaries[boundary].shape[0])):
            #         scaling_factor = max_random_magnitude[boundary]/tf.reduce_max(tf.abs(boundaries[boundary][i,:]))
            #         boundaries[boundary][i,:].assign(boundaries[boundary][i,:] * scaling_factor)
            # boundaries[boundary] = boundaries[boundary].numpy()
        else:
            boundaries[boundary] = tf.zeros((batch_size, boundary_lengths[boundary]), dtype = tf.keras.backend.floatx())
        if return_with_expanded_dims:
            if data_format == 'channels_first':
                boundaries[boundary] = tf.expand_dims(boundaries[boundary], axis = 1)
            else:
                boundaries[boundary] = tf.expand_dims(boundaries[boundary], axis = 2)
    return boundaries

def generate_dataset(batch_size, output_shape, dx = 'random', boundaries = 'random', rhses = 'random', smoothness = 1, rhs_max_random_magnitude = 1.0, boundary_max_random_magnitude = {'left':1.0, 'top':1.0, 'right':1.0, 'bottom': 1.0}, nonzero_boundaries = ['left', 'right', 'bottom', 'top'], solver_method = 'multigrid', return_rhs = True, return_boundaries = False, return_dx = False):
    '''
    Generates Poisson equation RHS-solution pairs with 'random' RHS functions.
    
    output_shape: shape of the outputs
    h: grid spacing of the outputs
    boundaries: boundary conditions of the outputs; see poisson_RHS documentation
    initial_smoothness and smoothness_levels: See documentation for __main__
    batch_size: No of random pairs to generate per smoothness level
    max_random_magnitude: See documentation for generate_random_RHS
    
    Outputs a tf.Tensor of shape (batch_size * smoothness_levels, n[0], n[1])
    '''
    if rhses == 'random':
        rhses = generate_random_RHS(batch_size, smoothness, output_shape, max_random_magnitude = rhs_max_random_magnitude)
        # rhses = generate_random_RHS, zip(itertools.repeat(batch_size, smoothness_levels), itertools.cycle(np.arange(initial_smoothness, initial_smoothness + smoothness_levels)), itertools.repeat(output_shape), itertools.repeat(tf.image.ResizeMethod.BICUBIC) ,itertools.repeat(rhs_max_random_magnitude)))), axis=0)
    elif rhses == 'zero':
        rhses = np.zeros([batch_size,1] + list(output_shape))
    
    if boundaries == 'random':
        boundaries = generate_random_boundaries(output_shape, batch_size = batch_size, max_random_magnitude = boundary_max_random_magnitude, return_with_expanded_dims = True, nonzero_boundaries = nonzero_boundaries)
    elif boundaries == 'zero' or boundaries == 'homogeneous':
        boundaries = generate_random_boundaries(output_shape, batch_size = batch_size, return_with_expanded_dims = True, nonzero_boundaries = [])

    if dx == 'random':
        dx = (0.1+np.random.rand(batch_size))*0.1

    if not callable(solver_method):
        if solver_method == 'cholesky':
            solver_method = cholesky_poisson_solve
        elif solver_method == 'multigrid':
            solver_method = multigrid_poisson_solve
        else:
            raise(ValueError('solver_method must be a function or one of cholesky or multigrid'))
        
    out = [solver_method(rhses, boundaries,dx)]
    if return_rhs:
        out.append(rhses)
    if return_boundaries:
        out.append(boundaries)
    if return_dx:
        out.append(dx)
    return tuple(out)

if __name__ == '__main__':
    from Boundary import Boundary1D
    import h5py, os, sys, time
    try:
        opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.925)
        conf = tf.ConfigProto(gpu_options=opts)
        tf.enable_eager_execution(config=conf)
    except:
        pass
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

