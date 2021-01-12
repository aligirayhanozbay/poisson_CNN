'''
TF 2.0 U-Net implementation based on the following Github repository:
https://github.com/jakeret/unet

@article{akeret2017radio,
  title={Radio frequency interference mitigation using deep convolutional neural networks},
  author={Akeret, Joel and Chang, Chihway and Lucchi, Aurelien and Refregier, Alexandre},
  journal={Astronomy and Computing},
  volume={18},
  pages={35--39},
  year={2017},
  publisher={Elsevier}
}
'''

from typing import Optional, Union, Callable, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam

from ..layers import deconvupscale

def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


def _get_kernel_initializer(filters, kernel_size):
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    return TruncatedNormal(stddev=stddev)


class ConvBlock(layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, dropout_rate, padding, activation, **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.dropout_rate=dropout_rate
        self.padding=padding
        self.activation=activation

        filters = _get_filter_count(layer_idx, filters_root)
        self.conv2d_1 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding=padding)
        self.dropout_1 = layers.Dropout(rate=dropout_rate)
        self.activation_1 = layers.Activation(activation)

        self.conv2d_2 = layers.Conv2D(filters=filters,
                                      kernel_size=(kernel_size, kernel_size),
                                      kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                      strides=1,
                                      padding=padding)
        self.dropout_2 = layers.Dropout(rate=dropout_rate)
        self.activation_2 = layers.Activation(activation)

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        x = self.conv2d_1(x)

        if training:
            x = self.dropout_1(x)
        x = self.activation_1(x)
        x = self.conv2d_2(x)

        if training:
            x = self.dropout_2(x)

        x = self.activation_2(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    dropout_rate=self.dropout_rate,
                    padding=self.padding,
                    activation=self.activation,
                    **super(ConvBlock, self).get_config(),
                    )


class UpconvBlock(layers.Layer):

    def __init__(self, layer_idx, filters_root, kernel_size, pool_size, padding, activation, **kwargs):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.padding=padding
        self.activation=activation

        filters = _get_filter_count(layer_idx + 1, filters_root)
        self.upconv = layers.Conv2DTranspose(filters // 2,
                                             kernel_size=(pool_size, pool_size),
                                             kernel_initializer=_get_kernel_initializer(filters, kernel_size),
                                             strides=pool_size, padding=padding)

        self.activation_1 = layers.Activation(activation)

    def call(self, inputs, **kwargs):
        x = inputs
        x = self.upconv(x)
        x = self.activation_1(x)
        return x

    def get_config(self):
        return dict(layer_idx=self.layer_idx,
                    filters_root=self.filters_root,
                    kernel_size=self.kernel_size,
                    pool_size=self.pool_size,
                    padding=self.padding,
                    activation=self.activation,
                    **super(UpconvBlock, self).get_config(),
                    )

class CropConcatBlock(layers.Layer):

    def call(self, x, down_layer, **kwargs):
        x1_shape = tf.shape(down_layer)
        x2_shape = tf.shape(x)

        if tf.keras.backend.image_data_format() == 'channels_last':
            height_diff = (x1_shape[1] - x2_shape[1]) // 2
            width_diff = (x1_shape[2] - x2_shape[2]) // 2
            down_layer_cropped = down_layer[:,
                                        height_diff: (x1_shape[1] - height_diff),
                                        width_diff: (x1_shape[2] - width_diff),
                                            :]
        else:
            height_diff = (x1_shape[2] - x2_shape[2]) // 2
            width_diff = (x1_shape[3] - x2_shape[3]) // 2
            down_layer_cropped = down_layer[:,:,
                                        height_diff: (x1_shape[2] - height_diff),
                                        width_diff: (x1_shape[3] - width_diff)]

        x = tf.concat([down_layer_cropped, x], axis=-1 if tf.keras.backend.image_data_format() == 'channels_last' else 1)
        return x




class UNetModel(tf.keras.models.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.data_format = tf.keras.backend.image_data_format()

    # def call(self, inp):
    #     print(inp.shape)
    #     if self.data_format == 'channels_first':
    #         inp = tf.transpose(inp, [0,2,3,1])
    #         out = super().call(inp)
    #         return tf.transpose(out, [0,3,1,2])
    #     else:
    #         return super().call(inp)
        
    def train_step(self, batch):
        # import pdb
        # pdb.set_trace()
        inputs, ground_truth = batch

        rhses, dx = inputs

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            pred = self(rhses)
            loss = self.loss_fn(y_true = ground_truth, y_pred = pred, rhs = rhses, dx = tf.concat([dx,dx],1))
        # print(loss)
        grads = tape.gradient(loss,self.trainable_variables)
        grad_L2 = (tf.reduce_mean([tf.reduce_sum(x**2) for x in grads]))**0.5
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'loss' : loss, 'mse': tf.reduce_mean((pred - ground_truth)**2), 'grad L2 norm': grad_L2, 'lr': self.optimizer.learning_rate}
    def compile(self, loss, optimizer):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss

def UNet(
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        layer_depth: int = 5,
        filters_root: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_rate: int = 0.5,
        padding:str="valid",
        activation:Union[str, Callable]="relu",
        final_activation:Union[str,Callable]="linear") -> Model:
    """
    Constructs a U-Net model
    :param nx: (Optional) image size on x-axis
    :param ny: (Optional) image size on y-axis
    :param in_channels: number of input channels of the input tensors
    :param out_channels: number of channels of the output
    :param layer_depth: total depth of unet
    :param filters_root: number of filters in top unet layer
    :param kernel_size: size of convolutional layers
    :param pool_size: size of maxplool layers
    :param dropout_rate: rate of dropout
    :param padding: padding to be used in convolutions
    :param activation: activation to be used
    :return: A TF Keras model
    """
    inpshape = (nx, ny, in_channels) if tf.keras.backend.image_data_format() == 'channels_last' else (in_channels,nx,ny)
    inputs = Input(shape=inpshape, name="inputs")

    x = inputs
    contracting_layers = {}
    shrs = {}

    conv_params = dict(filters_root=filters_root,
                       kernel_size=kernel_size,
                       dropout_rate=dropout_rate,
                       padding=padding,
                       activation=activation)

    for layer_idx in range(0, layer_depth - 1):
        x = ConvBlock(layer_idx, **conv_params)(x)
        contracting_layers[layer_idx] = x
        shr = tf.keras.layers.Lambda(lambda x: tf.shape(x), output_shape = (4,))(x)
        shrs[layer_idx] = shr
        x = layers.MaxPooling2D((pool_size, pool_size), padding = 'same', data_format = tf.keras.backend.image_data_format())(x)
        

    x = ConvBlock(layer_idx + 1, **conv_params)(x)

    for layer_idx in range(layer_idx, -1, -1):
        # x = UpconvBlock(layer_idx,
        #                 filters_root,
        #                 kernel_size,
        #                 pool_size,
        #                 padding,
        #                 activation)(x)
        x = deconvupscale(
            upsample_ratio = pool_size,
            filters = _get_filter_count(layer_idx + 1, filters_root)//2,
            kernel_size = pool_size,
            data_format = tf.keras.backend.image_data_format(),
            activation = tf.keras.layers.Activation(activation),
            dimensions = 2
        )([x,shrs[layer_idx]])
        #_asd = x([tf.random.uniform((10,100,100,2)), [10,200,200,_get_filter_count(layer_idx + 1, filters_root)//2]])
        #print(_asd.shape)
        #quit()
        x = CropConcatBlock()(x, contracting_layers[layer_idx])
        x = ConvBlock(layer_idx, **conv_params)(x)

    x = layers.Conv2D(filters=out_channels,
                      kernel_size=(1, 1),
                      kernel_initializer=_get_kernel_initializer(filters_root, kernel_size),
                      strides=1,
                      padding=padding)(x)

    x = layers.Activation(activation)(x)
    outputs = layers.Activation(final_activation, name="outputs")(x)
    model = UNetModel(inputs, outputs, name="unet")

    return model



# def finalize_model(model: Model,
#                    loss: Optional[Union[Callable, str]]=losses.categorical_crossentropy,
#                    optimizer: Optional= None,
#                    metrics:Optional[List[Union[Callable,str]]]=None,
#                    dice_coefficient: bool=True,
#                    auc: bool=True,
#                    mean_iou: bool=True,
#                    **opt_kwargs):
#     """
#     Configures the model for training by setting, loss, optimzer, and tracked metrics
#     :param model: the model to compile
#     :param loss: the loss to be optimized. Defaults to `categorical_crossentropy`
#     :param optimizer: the optimizer to use. Defaults to `Adam`
#     :param metrics: List of metrics to track. Is extended by `crossentropy` and `accuracy`
#     :param dice_coefficient: Flag if the dice coefficient metric should be tracked
#     :param auc: Flag if the area under the curve metric should be tracked
#     :param mean_iou: Flag if the mean over intersection over union metric should be tracked
#     :param opt_kwargs: key word arguments passed to default optimizer (Adam), e.g. learning rate
#     """

#     if optimizer is None:
#         optimizer = Adam(**opt_kwargs)

#     if metrics is None:
#         metrics = ['categorical_crossentropy',
#                    'categorical_accuracy',
#                    ]

#     if mean_iou:
#         metrics += [unet.metrics.mean_iou]

#     if dice_coefficient:
#         metrics += [unet.metrics.dice_coefficient]

#     if auc:
#         metrics += [tf.keras.metrics.AUC()]

#     model.compile(loss=loss,
#                   optimizer=optimizer,
#                   metrics=metrics,
#                   )

if __name__=='__main__':
    tf.config.run_functions_eagerly(True)
    tf.keras.backend.set_image_data_format('channels_first')
    z = UNet(in_channels = 1, nx = None, ny = None, layer_depth = 4, padding='same')
    z.summary()
    for nx in range(97,105):
        print('=====')
        dshape = (10,nx,nx,1) if tf.keras.backend.image_data_format() == 'channels_last' else (10,1,nx,nx)
        m = tf.random.uniform(dshape)
        print(z(m).shape)
    z.summary()
    res = z(m)
    import pdb
    pdb.set_trace()
