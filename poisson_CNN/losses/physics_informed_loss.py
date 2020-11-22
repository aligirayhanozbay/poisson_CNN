import tensorflow as tf

from ..dataset.generators.reverse import choose_conv_method
from ..dataset.utils import build_fd_coefficients, compute_domain_sizes

class linear_operator_loss:
    def __init__(self, stencil_sizes, orders, ndims = None, data_format = 'channels_first', normalize = False, inputs_have_max_domain_size_squared_normalization = False):
        if ndims is None:
            try:
                self.ndims = len(stencil_sizes)
            except:
                try:
                    self.ndims = len(orders)
                except:
                    raise(ValueError('If ndims is not supplied, one of stencil_sizes or orders must be a list containing as many elements as there are dimensions'))
        else:
            self.ndims = ndims
        self.stencil = tf.cast(tf.constant(build_fd_coefficients(stencil_sizes, orders, ndims)),tf.keras.backend.floatx())
        self.conv_method = choose_conv_method(self.ndims)
        self.data_format = data_format
        self.normalize = normalize
        self.inputs_have_max_domain_size_squared_normalization = inputs_have_max_domain_size_squared_normalization
        self.mse = lambda y_true,y_pred: (y_true-y_pred)**2

    def get_rhs_indices(self, rhs_shape):
        lower = tf.convert_to_tensor(self.stencil.shape[1:])//2
        upper = (rhs_shape[2:] if self.data_format == 'channels_first' else rhs_shape[1:-1]) - lower

        if self.data_format == 'channels_first':
            return [Ellipsis] + [slice(lower[k],upper[k]) for k in range(self.ndims)]
        else:
            return [Ellipsis] + [slice(lower[k],upper[k]) for k in range(self.ndims)] + [slice(0,rhs_shape[-1])]

    @tf.function
    def __call__(self,rhs,solution,grid_spacings):
        if self.inputs_have_max_domain_size_squared_normalization:
            q = (tf.expand_dims(tf.reduce_max(compute_domain_sizes(grid_spacings, tf.shape(solution)[2:] if self.data_format == 'channels_first' else tf.shape(solution)[1:-1]),1),-1)/grid_spacings)**2
        else:
            q = 1/(grid_spacings**2)
        
        kernels = tf.einsum('i...,bi->b...',self.stencil,q)
        kernels = tf.reshape(kernels,tf.unstack(tf.shape(kernels)) + [1,1])
        rhs_computed = tf.map_fn(lambda x: self.conv_method(tf.expand_dims(x[0],0),x[1],data_format=self.data_format), (solution,kernels), dtype=tf.keras.backend.floatx())[:,0,...]
        if self.normalize:
            max_rhs_magnitudes = tf.map_fn(lambda x: tf.reduce_max(tf.abs(x)),rhs)
            squared_error = self.mse(rhs[self.get_rhs_indices(tf.shape(rhs))],rhs_computed)
            squared_error_normalized = tf.einsum('i...,i->i...', squared_error, 1/max_rhs_magnitudes**2)
            return tf.reduce_mean(squared_error_normalized)
        else:
            return tf.reduce_mean(self.mse(rhs[self.get_rhs_indices(tf.shape(rhs))],rhs_computed))
        
if __name__ == '__main__':
    tf.keras.backend.set_floatx('float64')
    loss_func = linear_operator_loss(5,2,ndims=2)
    rhs = tf.random.uniform((10,1,50,75))
    soln = tf.random.uniform((10,1,50,75))
    gs = tf.random.uniform((10,2))

    from ..dataset.generators.reverse import reverse_poisson_dataset_generator
    nmax = 1500
    nmin = 50
    dmax = (1e+0)*1/(nmin-1)
    dmin = (1e+0)*1/(nmax-1)
    dxrange = [(1e+0)*dmin,(1e+0)*dmax]
    ndims = 2
    grid_size_range = [[nmin,nmax] for _ in range(ndims)]
    cmax = 10
    cmin = 5
    ctrl_pt_range = [[cmin,cmax] for _ in range(ndims)]
    hbc = True
    rpdg = reverse_poisson_dataset_generator(10,10,grid_size_range,ctrl_pt_range,ctrl_pt_range,grid_spacings_range=dxrange,homogeneous_bc = hbc,normalize_domain_size = False)
    inp, out = rpdg.__getitem__()
    print(loss_func(inp[0],out,inp[-1]))
