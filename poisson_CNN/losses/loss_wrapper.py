import tensorflow as tf

from .physics_informed_loss import linear_operator_loss
from .integral_loss import integral_loss

class loss_wrapper(tf.keras.losses.Loss):
    def __init__(self, ndims, integral_loss_weight, integral_loss_config, physics_informed_loss_weight, physics_informed_loss_config, data_format = 'channels_first'):
        super().__init__()
        self.ndims = ndims
        self.integral_loss_weight = integral_loss_weight
        self.physics_informed_loss_weight = physics_informed_loss_weight
        self.data_format = data_format

        integral_loss_config['ndims'] = self.ndims
        integral_loss_config['data_format'] = self.data_format

        physics_informed_loss_config['ndims'] = self.ndims
        physics_informed_loss_config['data_format'] = self.data_format

        self.integral_loss = integral_loss(**integral_loss_config)
        self.physics_informed_loss = linear_operator_loss(**physics_informed_loss_config)

    @tf.function
    def call(self,y_true,y_pred):
        print(y_pred)
        soln_pred, rhs, dx = y_pred

        loss = self.physics_informed_loss_weight * self.physics_informed_loss(rhs,soln_pred,dx)
        loss += self.integral_loss_weight * self.integral_loss(y_true,soln_pred)

        return loss

if __name__ == '__main__':
    physics_informed_loss_config = {
		"stencil_sizes":[5,5],
		"orders":2,
		"normalize":False
	    }
    integral_loss_config = {
		"n_quadpts":20,
		"Lp_norm_power":2
	    }
    loss = loss_wrapper(2,1.0,integral_loss_config,1.0,physics_informed_loss_config)
    soln = tf.random.uniform((10,1,200,200))
    actual_soln = 2*soln
    rhs = tf.random.uniform((10,1,200,200))
    dx = tf.random.uniform((10,2))
    print(loss(actual_soln, [soln,rhs,dx]))
