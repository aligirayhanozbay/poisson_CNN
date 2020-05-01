import tensorflow as tf

from ..dataset.generators.reverse import choose_conv_method

class linear_operator_loss:
    def __init__(self,kernel, data_format = 'channels_first'):
        self.ndims = len(kernel.shape)-2
        self.kernel = kernel
        self.conv_method = choose_conv_method(self.ndims)
        self.data_format = data_format
        self.mse = tf.keras.losses.MeanSquaredError()

    def get_rhs_indices(self, rhs_shape):
        lower = tf.convert_to_tensor(self.kernel.shape[:-2])//2
        upper = (rhs_shape[2:] if self.data_format == 'channels_first' else rhs_shape[1:-1]) - lower

        if self.data_format == 'channels_first':
            return [Ellipsis] + [slice(lower[k],upper[k]) for k in range(self.ndims)]
        else:
            return [Ellipsis] + [slice(lower[k],upper[k]) for k in range(self.ndims)] + [slice(0,rhs_shape[-1])]

    @tf.function
    def __call__(self,rhs,solution):
        reconstructed_rhs = self.conv_method(solution, self.kernel, data_format = 'channels_first')
        return self.mse(reconstructed_rhs, rhs[self.get_rhs_indices(tf.shape(rhs))])
        
if __name__ == '__main__':
    loss_func = linear_operator_loss(tf.random.uniform((5,7,1,3)))
    rhs = tf.random.uniform((10,1,50,75))
    soln = tf.random.uniform((10,1,50,75))
    print(loss_func(rhs,soln).shape)
