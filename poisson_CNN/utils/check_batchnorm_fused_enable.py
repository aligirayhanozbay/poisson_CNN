import tensorflow as tf

def check_batchnorm_fused_enable():
    floatx = tf.keras.backend.floatx()
    batchnorm_fused_allowed_datatypes = ['float16', 'float32', 'bfloat16']
    if floatx in batchnorm_fused_allowed_datatypes:
        return True
    else:
        return False
