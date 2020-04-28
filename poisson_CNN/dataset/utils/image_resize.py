import tensorflow as tf
import multiprocessing

@tf.function
def image_resize(image, newshape, data_format = 'channels_first', resize_method = tf.image.ResizeMethod.BICUBIC):
    '''
    Helper function to resize images on the CPU in parallel
    '''
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
