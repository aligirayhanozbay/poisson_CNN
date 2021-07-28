import tensorflow as tf

@tf.function
def compute_domain_sizes(dx, domain_shape):
        domain_sizes = tf.einsum('ij,j->ij', dx, tf.cast(domain_shape-1,dx.dtype))
        return domain_sizes
