import tensorflow as tf
import numpy as np
from ..dataset.utils import equal_split_tensor_slice
    
class SpatialPyramidPool(tf.keras.layers.Layer):
    def __init__(self, levels, ndims, data_format = 'channels_first', pooling_type = 'average', padding = 'same', padding_mode = 'constant', constant_padding_value = 0.0):
        super().__init__()
        self.ndims = ndims
        self.data_format = data_format

        if pooling_type == 'average':
            self.pooling_func = tf.reduce_mean
        elif pooling_type == 'max':
            self.pooling_func = tf.reduce_max
        
        for k in range(len(levels)):
            print(levels[k])
            if isinstance(levels[k],int):
                levels[k] = [levels[k] for _ in range(ndims)]
            elif len(levels[k]) == 1:
                levels[k] = [levels[k][0] for _ in range(ndims)]
            elif len(levels[k]) != self.ndims:
                raise(ValueError('Each SPP level must have a pool size with ndims or 1 element(s). Got ' + str(len(levels[k]))))
        self.nlevels = len(levels)
        print(levels)
        self.levels = tf.constant(levels)


    @tf.function
    def get_pool_component(self,tensor,level,component_idx):
        bin_values = equal_split_tensor_slice(tensor, component_idx, level, ndims = self.ndims)
        pooled_values = tf.map_fn(self.pooling_func,bin_values)
        return pooled_values
        
    @tf.function
    def call(self,inp):
        inpshape = tf.shape(inp)

        results = []
        for k in range(self.nlevels):
            level = self.levels[k]
            bin_idx = tf.stack(tf.meshgrid(*[tf.range(0,level[k]) for k in range(self.ndims)],indexing='ij'),-1)
            bin_idx = tf.reshape(bin_idx,[-1,self.ndims])
            pooled_values = tf.map_fn(lambda x: self.get_pool_component(inp,level,x), bin_idx, dtype = inp.dtype)
            pooled_values = tf.transpose(pooled_values,(1,0))
            results.append(pooled_values)

        return tf.concat(results,1)

if __name__ == '__main__':
    mod = SpatialPyramidPool([[2,2,2],3],3)
    inp = tf.random.uniform((10,5,2000,20 00))
    q=mod(inp)
    import time
    t0 = time.time()
    for k in range(10):
        print(mod(inp).shape)
    print((time.time() - t0)/10)
        
            
        
