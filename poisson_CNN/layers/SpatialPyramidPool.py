import tensorflow as tf
import numpy as np

class SpatialPyramidPool(tf.keras.layers.Layer):
    def __init__(self, levels, data_format = 'channels_first', pooling_type = 'AVG' ,**kwargs):
        super().__init__(**kwargs)
        self.maxdims = 3
        self.data_format = data_format
        self.pooling_type = pooling_type
        self.levels = levels
        #self.data_format = data_format
        self.ndims = None
        #check if all SPP levels have the same number of dimensions, fix dimensions along which only ints were given
        level_dimensionalities = []
        for level in self.levels:
            try:
                level_dimensionalities.append(len(level))
            except:
                level_dimensionalities.append(0)
        level_dimensionalities = np.array(level_dimensionalities)
        if np.all(level_dimensionalities == 0):
            print('Spatial Pyramid Pooling Layer Warning: Input dimensionality can not be inferred. This can negatively impact performance.')
            return    
        elif np.any(level_dimensionalities[level_dimensionalities!=0] != level_dimensionalities[level_dimensionalities !=0][0]):
            raise(ValueError('Dimension number mismatch in the provided level output shapes'))
        else:
            self.ndims = level_dimensionalities[level_dimensionalities != 0][0]
            for i in range(len(self.levels)):
                if isinstance(self.levels[i], int):
                    s = int(self.levels[i])
                    self.levels[i] = [s for i in range(self.ndims)]
            if self.ndims > self.maxdims:
                raise(ValueError('ndims must be between 1 and ' + str(self.maxdims)))
            elif self.data_format == 'channels_first':
                self.data_format = 'NC' + 'DHW'[(self.maxdims-self.ndims):]
            else:
                self.data_format = 'N' + 'DHW'[(self.maxdims-self.ndims):] + 'C'
    #@tf.function
    def call(self, inp):
        if self.ndims == None: #build the pooling levels based on input shape if not already determined
            ndims = len(inp.shape) - 2
            levels = []
            for i in range(len(self.levels)):
                s = int(self.levels[i])
                levels.append([s for i in range(ndims)])
            if ndims > self.maxdims:
                raise(ValueError('ndims must be between 1 and ' + str(self.maxdims)))
            elif self.data_format == 'channels_first':
                data_format = 'NC' + 'DHW'[(self.maxdims-ndims):]
            else:
                data_format = 'N' + 'DHW'[(self.maxdims-ndims):] + 'C'
        else:
            data_format = self.data_format
            levels = self.levels
            ndims = self.ndims
        try:
            if data_format[1] == 'C':
                strides = [tf.cast(tf.math.floor(inp.shape[2+k]/levels[0][k]), dtype = tf.int32) for k in range(ndims)]
                windowshape = [strides[k] + (inp.shape[2+k] % levels[0][k]) for k in range(ndims)]
            else:
                strides = [tf.cast(tf.math.floor(inp.shape[2+k]/levels[0][k]), dtype = tf.int32) for k in range(ndims)]
                windowshape = [strides[k] + (inp.shape[1+k] % levels[0][k]) for k in range(ndims)]
            out = tf.keras.backend.reshape(tf.nn.pool(inp, window_shape=windowshape, strides=strides, pooling_type=self.pooling_type, padding = 'VALID', data_format=data_format), [inp.shape[0], -1])
            #out = tf.keras.backend.pool2d(inp, tuple(windowshape), padding = 'valid', data_format = 'channels_first', pool_mode = 'avg', strides = tuple(strides))

            for i in range(1,len(levels)):
                if data_format[1] == 'C':
                    strides = [tf.cast(tf.math.floor(inp.shape[2+k]/levels[i][k]), dtype = tf.int32) for k in range(ndims)]
                    windowshape = [strides[k] + (inp.shape[2+k] % levels[i][k]) for k in range(ndims)]
                else:
                    strides = [tf.cast(tf.math.floor(inp.shape[2+k]/levels[i][k]), dtype = tf.int32) for k in range(ndims)]
                    windowshape = [strides[k] + (inp.shape[1+k] % levels[i][k]) for k in range(ndims)]
                out = tf.concat([out, tf.reshape(tf.nn.pool(inp, window_shape=windowshape, strides=strides, pooling_type=self.pooling_type, padding = 'VALID', data_format=data_format), [inp.shape[0],-1])],1)
        except:
            print('spp init')
            if data_format[1] == 'C':
                out = tf.reduce_max(inp, axis = [k for k in range(2,2+ndims)])
            else:
                out = tf.reduce_max(inp, axis = [k for k in range(1,1+ndims)])
            levels = tf.constant(levels)
            out = tf.tile(out, [1,tf.reduce_sum(tf.reduce_prod(levels, axis = 1))])
        return out
