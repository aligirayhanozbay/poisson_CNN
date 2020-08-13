import tensorflow as tf

from .physics_informed_loss import linear_operator_loss
from .integral_loss import integral_loss

class loss_wrapper(tf.keras.losses.Loss):
    def __init__(self, ndims, integral_loss_weight, integral_loss_config, physics_informed_loss_weight, physics_informed_loss_config, data_format = 'channels_first', mse_loss_weight = 0.0, mae_loss_weight = 0.0, scale_sample_loss_by_target_peak_magnitude = False, global_batch_size = None):
        '''
        Provides a convenient way to bundle 4 types of loss together: integral loss, physics informed loss, MAE and MSE.

        Init arguments:
        -ndims: # of spatial dimensions
        -integral_loss_weight: float. Weight to assign to the integral loss.
        -integral_loss_config: dict. Init arguments to be supplied to integral loss.
        -physics_informed_loss_weight: float. Weight to assign to the physics informed (PI) loss.
        -physics_informed_loss_config: dict. Init arguments to be supplied to PI loss.
        -data_format: str. "channels_first" or "channels_last" - same as tf.keras.
        -mse_loss weight: float. Weight to assign to the MSE loss.
        -mae_loss_weight: float. Weight to assign to the MAE loss.
        -scale_sample_loss_by_target_peak_magnitude: bool. If set to true, each loss value will be scaled by 1/max(abs(sample)).
        -global_batch_size: int. Batch size to use across all devices. Necessary for compatibility for distributed training with tf.keras.models.Model.fit() method with. Do not use otherwise.
        '''
        super().__init__()
        self.ndims = ndims
        self.integral_loss_weight = integral_loss_weight
        self.physics_informed_loss_weight = physics_informed_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.mae_loss_weight = mae_loss_weight
        self.data_format = data_format
        self.scale_sample_loss_by_target_peak_magnitude = scale_sample_loss_by_target_peak_magnitude

        integral_loss_config['ndims'] = self.ndims
        integral_loss_config['data_format'] = self.data_format

        physics_informed_loss_config['ndims'] = self.ndims
        physics_informed_loss_config['data_format'] = self.data_format

        self.integral_loss = integral_loss(reduce_results = True, **integral_loss_config)
        self.physics_informed_loss = linear_operator_loss(**physics_informed_loss_config)
        self.mse_loss = lambda y_true, y_pred: tf.reduce_mean((y_true - y_pred)**2) #tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.mae_loss = lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred))#tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.global_batch_size = global_batch_size

    @tf.function
    def _compute_supervised_loss_and_scale_by_target_peak_magnitude(self, loss_func, y_true, y_pred, peak_magnitude_scaling_power = tf.constant(1.0, dtype = tf.keras.backend.floatx()), target_peak_magnitudes = None):
        global_batch_size = tf.shape(y_true)[0] if self.global_batch_size is None else self.global_batch_size
        loss_per_sample = tf.map_fn(lambda x: loss_func(x[0],x[1]), [y_true, y_pred], dtype=y_true.dtype)
        sample_weights = 1/(target_peak_magnitudes**peak_magnitude_scaling_power) if self.scale_sample_loss_by_target_peak_magnitude else tf.ones(tf.shape(loss_per_sample), dtype = loss_per_sample.dtype)
        loss_val = tf.reduce_sum(sample_weights * loss_per_sample) / tf.cast(global_batch_size, tf.keras.backend.floatx())
        return loss_val

    @tf.function
    def __call__(self,y_true,y_pred,rhs,dx):

        loss = tf.constant(0.0,dtype=tf.keras.backend.floatx())

        target_peak_magnitudes = tf.map_fn(tf.reduce_max,tf.abs(y_true)) if self.scale_sample_loss_by_target_peak_magnitude else None
        
        if self.mse_loss_weight != 0.0:
            loss += self.mse_loss_weight * self._compute_supervised_loss_and_scale_by_target_peak_magnitude(self.mse_loss, y_true, y_pred, peak_magnitude_scaling_power = 2.0, target_peak_magnitudes = target_peak_magnitudes)
            #loss += self.mse_loss_weight * self.mse_loss(y_true, y_pred)
        if self.mae_loss_weight != 0.0:
            loss += self.mae_loss_weight * self._compute_supervised_loss_and_scale_by_target_peak_magnitude(self.mae_loss, y_true, y_pred, peak_magnitude_scaling_power = 1.0, target_peak_magnitudes = target_peak_magnitudes)
            #loss += self.mae_loss_weight * self.mae_loss(y_true, y_pred)
        if self.physics_informed_loss_weight != 0.0:
            loss += self.physics_informed_loss_weight * self.physics_informed_loss(rhs,y_pred,dx)
        if self.integral_loss_weight != 0.0:
            loss += self.integral_loss_weight * self._compute_supervised_loss_and_scale_by_target_peak_magnitude(self.integral_loss, tf.expand_dims(y_true,1), tf.expand_dims(y_pred,1), peak_magnitude_scaling_power = self.integral_loss.Lp_norm_power, target_peak_magnitudes = target_peak_magnitudes)
            #loss += tf.reduce_mean(self.integral_loss_weight * self.integral_loss(y_true,[y_pred,dx]))

        return loss

if __name__ == '__main__':
    physics_informed_loss_config = {
		"stencil_sizes":[5,5],
		"orders":2,
		"normalize":False
	    }
    integral_loss_config = {
		"n_quadpts":5,
		"Lp_norm_power":2
	    }
    loss = loss_wrapper(2,1.0,integral_loss_config,1.0,physics_informed_loss_config)
    soln = tf.random.uniform((10,1,200,200))
    actual_soln = 2*soln
    rhs = tf.random.uniform((10,1,200,200))
    dx = tf.random.uniform((10,2))
    print(loss(actual_soln,soln,rhs,dx))
