import tensorflow as tf
import math
import string

from .reverse import handle_grid_parameters_range, process_normalizations, polynomials_and_their_2nd_derivatives, reverse_poisson_dataset_generator
from ..utils import *
from ...utils import choose_conv_method

class reverse_poisson_dataset_generator_homogeneous_neumann(reverse_poisson_dataset_generator):
    def __init__(self, batch_size, batches_per_epoch, random_output_shape_range, fourier_coeff_grid_size_range, grid_spacings_range = None, ndims = None, return_rhses = True, return_dx = True, normalizations = None, uniform_grid_spacing = False):
        self.batch_size = batch_size
        if ndims is None:
            for k in [random_output_shape_range,fourier_coeff_grid_size_range]:
                try:
                    ndims = len(k)
                except:
                    pass
        self.ndims = ndims
        self.batches_per_epoch = batches_per_epoch
        self.homogeneous_bc = False

        self.grid_spacings_range = handle_grid_parameters_range(grid_spacings_range, ndims, tf.keras.backend.floatx())
        self.random_output_shape_range = handle_grid_parameters_range(random_output_shape_range, ndims, tf.int32)
        self.fourier_coeff_grid_size_range = handle_grid_parameters_range(fourier_coeff_grid_size_range, ndims, tf.int32)

        self.return_rhses = return_rhses
        self.return_dx = return_dx
        self.return_boundaries = False
        
        self.normalizations = process_normalizations(normalizations)
        self.uniform_grid_spacing = uniform_grid_spacing

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
        solns, coeffs = tf.map_fn(lambda x: generate_smooth_function(self.ndims,x[0],x[1],homogeneous_bc = False, homogeneous_neumann_bc = True, normalize = False, return_coefficients = True, coefficients_return_shape = tf.reduce_max(n_coefficients,0)),(output_shape,n_coefficients),dtype=(tf.keras.backend.floatx(),tf.keras.backend.floatx()))
        return tf.expand_dims(solns,1), coeffs, grid_spacings

    def __getitem__(self, idx = 0):
        #fourier component
        solns_fourier, soln_coeffs_fourier, grid_spacings_fourier = self.generate_soln_fourier()
        rhses_fourier, domain_sizes = self.generate_rhses_fourier(soln_coeffs_fourier, tf.shape(solns_fourier)[2:], grid_spacings_fourier)
        
        #sum components
        solns = solns_fourier
        rhses = rhses_fourier
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
    rpdg = reverse_poisson_dataset_generator_homogeneous_neumann(batch_size = 5, batches_per_epoch = 5, random_output_shape_range = grid_size_range, fourier_coeff_grid_size_range = ctrl_pt_range, grid_spacings_range = dxrange, ndims = 2, return_rhses = True, return_dx = True, normalizations = normalizations, uniform_grid_spacing = uniform_grid_spacing)
    inp, out = rpdg.__getitem__()

    if uniform_grid_spacing:
        inp[1] = tf.concat([inp[1] for _ in range(ndims)],1)
    from ...losses import linear_operator_loss
    loss_fn = linear_operator_loss(stencil_sizes = 5, orders = 2, ndims = 2, data_format = 'channels_first')
    loss_val = loss_fn(inp[0],out,inp[1])
    print(loss_val)
    import pdb
    pdb.set_trace()
    
