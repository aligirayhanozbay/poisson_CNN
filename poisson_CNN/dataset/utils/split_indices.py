import tensorflow as tf

@tf.function
def split_indices(dim_length,sections):
    '''
    Provides indices to split a dim_length size array into several roughly equally sized bins

    Inputs:
    -dim_length: int or tf.Constant. Size of the array
    -sections: Number of sections to create

    Outputs:
    -section_indices: tf.Tensor of shape (sections,) (Pythonic) indices where each bin should start/end. E.g. for dim_length 229 and sections we get [0 58 115 172 229] so bin 0 should be x[0:58], bin 1 should be [58:115] and so forth.
    '''
    elements_per_section = dim_length // sections
    extras = dim_length % sections
    
    sections_with_extras = tf.expand_dims(elements_per_section+1,0)
    sections_with_extras = tf.tile(sections_with_extras,[extras])
    
    sections_without_extras = tf.expand_dims(elements_per_section,0)
    sections_without_extras = tf.tile(sections_without_extras,[sections-extras])
    
    section_sizes = tf.concat([[0], sections_with_extras,sections_without_extras],0)
    
    return tf.math.cumsum(section_sizes)

if __name__ == '__main__':
    d = tf.constant(229)
    s = tf.constant(4)
    print(split_indices(d,s))
