import tensorflow as tf
import numpy as np
from collections.abc import Iterable

def change_value(tensor,value,idx):
    tensor = np.array(tensor)
    if not isinstance(idx,Iterable):
        idx = (int(idx),)
    tensor[tuple(idx)] = value
    return tensor

@tf.function
def assign_to_tensor_index(tensor,value,idx):
    value = tf.convert_to_tensor(value)
    input_tensor_as_variable = lambda: tf.Variable(initial_value=tensor)
    input_tensor_as_variable[idx].assign(value)
    return tf.convert_to_tensor(input_tensor_as_variable)
    #return tf.reshape(tf.numpy_function(change_value,[tensor,value,idx],tensor.dtype),tensor.shape)

if __name__=='__main__':

    b = tf.zeros((5,6,7,8))
    import time
    print('---assign_to_tensor_index unit test---')
    t0 = time.time()
    bnp = assign_to_tensor_index(b,-1.0,[0,1,2,3])
    t1 = time.time()
    print('numpy version took ' + str(t1-t0) + ' seconds')
    if bnp[0,1,2,3] == -1.0 and tf.reduce_sum(bnp) == -1.0:
        print('Value assignment successful')
    else:
        print('Value assignment unsuccessful (wrong value)')


    
