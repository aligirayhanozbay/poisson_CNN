import tensorflow as tf

def check_batchnorm_fused_enable(ndims = 2):
    floatx = tf.keras.backend.floatx()
    batchnorm_fused_allowed_datatypes = ['float16', 'float32', 'bfloat16']
    batchnorm_fused_allowed_ndims = [2]
    if floatx in batchnorm_fused_allowed_datatypes and ndims in batchnorm_fused_allowed_ndims:
        return True
    else:
        return False
