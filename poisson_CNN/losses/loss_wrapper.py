import tensorflow as tf

from .physics_informed_loss import linear_operator_loss
from .integral_loss import integral_loss

class loss_wrapper:
    def __init__(self, ndims, integral_loss_weight, integral_loss_config, physics_informed_loss_weight, physics_informed_loss_config, data_format = 'channels_first'):
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
    def __call__(self,y_true,y_pred):
        pass
