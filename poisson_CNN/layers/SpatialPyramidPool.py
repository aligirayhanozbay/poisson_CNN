import tensorflow as tf
import numpy as np
from ..dataset.utils import equal_split_tensor_slice
    
class SpatialPyramidPool(tf.keras.layers.Layer):
    def __init__(self, levels, ndims, data_format = 'channels_first', pooling_type = 'average', receive_padded_values = False):
        super().__init__()
        self.ndims = ndims
        self.data_format = data_format

        if pooling_type.lower() == 'average' or pooling_type.lower() == 'avg':
            self.pooling_func = tf.reduce_mean
        elif pooling_type.lower() == 'max':
            self.pooling_func = tf.reduce_max
        
        for k in range(len(levels)):
            if isinstance(levels[k],int):
                levels[k] = [levels[k] for _ in range(ndims)]
            elif len(levels[k]) == 1:
                levels[k] = [levels[k][0] for _ in range(ndims)]
            elif len(levels[k]) != self.ndims:
                raise(ValueError('Each SPP level must have a pool size with ndims or 1 element(s). Got ' + str(len(levels[k]))))
        self.nlevels = len(levels)
        self.levels = tf.constant(levels)

        self.receive_padded_values = receive_padded_values


    @tf.function
    def get_pool_component(self,tensor,level,component_idx,pad_mask = None):
        if pad_mask is not None:
            non_padding_region_shape = tf.where(pad_mask)[-1]+1
            if self.data_format == 'channels_first':
                tensor = tf.transpose(tensor,[k+2 for k in range(self.ndims)] + [0,1])
            else:
                tensor = tf.transpose(tensor,[k+1 for k in range(self.ndims)] + [0,self.ndims+1])
            tensor = tf.boolean_mask(tensor,pad_mask)
            tensor = tf.reshape(tensor,tf.concat([non_padding_region_shape,tf.cast(tf.shape(tensor)[-2:],tf.int64)],0))
            if self.data_format == 'channels_first':
                tensor = tf.transpose(tensor,[self.ndims,self.ndims+1] + [k for k in range(self.ndims)])
            else:
                tensor = tf.transpose(tensor,[self.ndims] + [k for k in range(self.ndims)] + [self.ndims+1])
        bin_values = equal_split_tensor_slice(tensor, component_idx, level, ndims = self.ndims)
        pooled_values = tf.map_fn(self.pooling_func,bin_values)
        return pooled_values
        
    @tf.function
    def call(self,inp):
        if self.receive_padded_values:
            inp,pad_mask = inp

        inpshape = tf.shape(inp)

        results = []
        for k in range(self.nlevels):
            level = self.levels[k]
            bin_idx = tf.stack(tf.meshgrid(*[tf.range(0,level[k]) for k in range(self.ndims)],indexing='ij'),-1)
            bin_idx = tf.reshape(bin_idx,[-1,self.ndims])
            if self.receive_padded_values:
                pooled_values = tf.map_fn(lambda x: self.get_pool_component(inp,level,x,pad_mask), bin_idx, dtype = inp.dtype)
            else:
                pooled_values = tf.map_fn(lambda x: self.get_pool_component(inp,level,x), bin_idx, dtype = inp.dtype)
            pooled_values = tf.transpose(pooled_values,(1,0))
            results.append(pooled_values)

        return tf.concat(results,1)

if __name__ == '__main__':
    mod = SpatialPyramidPool([[2,2],3,5,8],2)
    inp = tf.random.uniform((10,5,2100,2000))
    q=mod(inp)
    import pdb
    pdb.set_trace()
    import time
    t0 = time.time()
    for k in range(10):
        print(mod(inp).shape)
    print((time.time() - t0)/10)

    mod = SpatialPyramidPool([[2,2],3],2,receive_padded_values = True)
    pad_mask_component0 = tf.concat([tf.ones((1850,)),tf.zeros((250,))],0)
    pad_mask_component1 = tf.concat([tf.ones((1558,)),tf.zeros((2000-1558,))],0)
    pad_mask = tf.cast(tf.einsum('i,j->ij',pad_mask_component0,pad_mask_component1),tf.bool)
    q=mod([inp,pad_mask])
    import time
    t0 = time.time()
    for k in range(10):
        print(mod([inp,pad_mask]).shape)
    print((time.time() - t0)/10)
    
        
            
        
