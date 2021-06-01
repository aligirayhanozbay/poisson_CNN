'''
Taken from https://www.tensorflow.org/tutorials/generative/pix2pix
'''

import tensorflow as tf

from ..layers import deconvupscale

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Pix2Pix_Generator(depth = 8, nx = None, ny = None, out_channels = 1, in_channels = 1, upsample_activation = tf.nn.leaky_relu):
    inpshape = (nx, ny, in_channels) if tf.keras.backend.image_data_format() == 'channels_last' else (in_channels,nx,ny)
    inputs = tf.keras.layers.Input(shape=inpshape)
    channels_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1
    
    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]
    
    shape_return_layers = [tf.keras.layers.Lambda(lambda u: tf.shape(u), output_shape = (4,)) for _ in range(len(down_stack))]
    up_stack = [
        deconvupscale(
            upsample_ratio = 2,
            filters = block.layers[0].filters,
            kernel_size = 4,
            data_format = tf.keras.backend.image_data_format(),
            activation = upsample_activation,
            dimensions = 2
        ) for block in reversed(down_stack)
    ]

    # up_stack = [
    #     upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    #     upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    #     upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    #     upsample(512, 4), # (bs, 16, 16, 1024)
    #     upsample(256, 4), # (bs, 32, 32, 512)
    #     upsample(128, 4), # (bs, 64, 64, 256)
    #     upsample(64, 4), # (bs, 128, 128, 128)
    # ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(out_channels, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
        
    skips = reversed(skips[:-1])
        
    # Upsampling and establishing the skip connections
    for up, shape, skip in zip(up_stack, shape_return_layers, skips):
        xshape = shape(skip)
        x = up([x, xshape])
        x = tf.keras.layers.Concatenate(axis = channels_axis)([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

if __name__ == '__main__':
    tf.keras.backend.set_image_data_format('channels_last')
    nx = 256
    ny = 256
    bsize = 3
    in_ch = 1
    out_ch = 1
    mod = Pix2Pix_Generator(nx = nx, ny = ny, out_channels = out_ch, in_channels = in_ch)
    mod.summary()

    inpshape = (bsize, nx, ny, in_ch) if tf.keras.backend.image_data_format() == 'channels_last' else (bsize, in_ch, nx, ny)
    inp = tf.random.uniform(inpshape)
    out = mod(inp)

