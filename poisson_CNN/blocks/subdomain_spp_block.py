import tensorflow as tf
import numpy as np

from ..layers import SpatialPyramidPool, metalearning_conv
from ..dataset.generators.reverse import choose_conv_method
from ..dataset.utils import equal_split_tensor_slice

class subdomain_spp_block(tf.keras.models.Model):
    def __init__(self,subdomain_partitions,ndims,spp_levels,spp_pooling_type = 'average',dense_layer_units = [16,16],dense_activations = tf.keras.activations.linear,metalearning_conv_args = None):
        super().__init__()
        self.ndims = ndims

        self.conv_layer = metalearning_conv(**metalearning_conv_args)
        self.data_format = self.conv_layer.data_format

        self.spp = SpatialPyramidPool(spp_levels, self.ndims, data_format = self.data_format, pooling_type = spp_pooling_type)

        self.subdomain_partitions = subdomain_partitions
        
        self.subdomain_partition_indices = [tf.range(0,self.subdomain_partitions[k]) for k in range(ndims)]
        self.subdomain_partition_indices = tf.stack(tf.meshgrid(*self.subdomain_partition_indices,indexing='ij'),-1)
        self.subdomain_partition_indices = tf.reshape(self.subdomain_partition_indices,[-1,ndims])
        self.subdomain_partition_indices = tf.unstack(self.subdomain_partition_indices,num=None,axis=0)

        if callable(dense_activations):
            dense_activations = [dense_activations for _ in range(len(dense_layer_units))]
        self.dense_layers = [tf.keras.layers.Dense(units,activation = act) for units,act in zip(dense_layer_units,dense_activations)]

        self.pre_spp_pad_value = -np.inf if spp_pooling_type == 'average' else 0.0 if spp_pooling_type == 'max' else None
        

    @tf.function
    def call(self,inp):
        conv_inp,domain_info = inp
        bsize = tf.shape(conv_inp)[0]
        out = self.conv_layer([conv_inp,domain_info])

        #0.68s per sample
        out = list(map(lambda x: equal_split_tensor_slice(out,x,self.subdomain_partitions,ndims=self.ndims),self.subdomain_partition_indices))
        #print([k.shape for k in out])
        out = list(map(self.spp, out))
        out = tf.stack(out,0)

        '''
        #0.98s per sample
        out,pad_masks = tf.map_fn(lambda x: equal_split_tensor_slice(out,x,self.subdomain_partitions,self.ndims,pad_to_equal_size=True,return_pad_mask=True,pad_value=0.0), self.subdomain_partition_indices, dtype = (out.dtype,tf.bool))

        out = tf.map_fn(self.spp,(out,pad_masks),dtype=out.dtype)
        '''

        for layer in self.dense_layers:
            out = layer(out)
        dense_features = tf.shape(out)[-1]

        out = tf.transpose(out,[1,2,0])
        out = tf.reshape(out, [bsize,dense_features] + self.subdomain_partitions)

        return out


if __name__ == '__main__':
    ndims = 2
    data_format = 'channels_first'
    mlc_args = {'previous_layer_filters': 1, 'filters': 4, 'kernel_size': 5, 'padding': 'same', 'padding_mode': 'SYMMETRIC','data_format': data_format, 'conv_activation':tf.nn.leaky_relu, 'dense_activations':tf.nn.leaky_relu, 'dimensions': 2}
    partitions = [6,5]
    spp_levels = [[2,2],3]
    spp_pooling_type = 'average'
    dense_layer_units = [16,24]
    dense_activations = mlc_args['dense_activations']
    ssb = subdomain_spp_block(partitions,ndims,spp_levels,spp_pooling_type=spp_pooling_type,dense_layer_units = dense_layer_units, dense_activations = dense_activations,metalearning_conv_args = mlc_args)
    inp = [tf.random.uniform((10,1,2500,2500)),tf.random.uniform((10,3))]
    print(ssb(inp).shape)
    #'''
    import time
    t0 = time.time()
    for k in range(10):
        print(ssb(inp).shape)
    print((time.time() - t0)/10)
    #'''

        
        

        
        
        
        
