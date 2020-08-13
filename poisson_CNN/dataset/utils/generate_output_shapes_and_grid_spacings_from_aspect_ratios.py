import tensorflow as tf

@tf.function
def generate_output_shapes_and_grid_spacings_from_aspect_ratios(aspect_ratios, random_output_shape_range, random_dx_range, constant_dx = False, samples = None):
    '''
    Generates output shapes and grid spacings given aspect ratio(s).

    Inputs:
    -aspect_ratios: int tf.Tensor of shape [batch_size,ndims] if constant_dx is False or [1,ndims] if constant_dx is True. The aspect ratios to use.
    -random_output_shape_range: int tf.Tensor of shape [ndims,2]. first spatial dimension will be chosen according to the values in this tensor. output shapes will never exceed the values across the max values (slice [:,1]) of this input, but they may be below the min values (slice [:,0]) if constant_dx is True
    -random_dx_range: float tf.Tensor of shape [ndims,2]. x dir grid spacing will be chosen according to the values in this tensor. random_dx_range across other dims may not be respected if constant_dx is False.
    -constant_dx: bool. If set to true, grid spacings will be identical in each dimension.
    -samples: int. No of samples to generate. Used only if constant_dx is True
    '''
    ndims = tf.shape(aspect_ratios)[1]+1
    
    random_output_shape_range = tf.convert_to_tensor(random_output_shape_range)
    nx_min = tf.cast(random_output_shape_range[0][0],tf.keras.backend.floatx())-0.4999999999
    nx_max = tf.cast(random_output_shape_range[0][1],tf.keras.backend.floatx())+0.4999999999
    nx = tf.cast(tf.math.round(tf.random.uniform((1,))*(nx_max-nx_min)+tf.cast(random_output_shape_range[0][0],tf.keras.backend.floatx())),tf.int32)[0]
    
    if constant_dx:#aspect ratio cant vary with constant dx and constant domain shape.
        dx = tf.tile(tf.random.uniform((samples,1))*(random_dx_range[0][1] - random_dx_range[0][0]) + random_dx_range[0][0],[1,ndims])
        npts_other_dimensions = tf.cast(tf.cast(nx,aspect_ratios.dtype)/aspect_ratios[0],tf.int32)
    else:
        dx = (random_dx_range[0][1] - random_dx_range[0][0]) * tf.random.uniform((tf.shape(aspect_ratios)[0],1),dtype=tf.keras.backend.floatx()) + random_dx_range[0][0]
        Lx = (tf.cast(nx-1,tf.keras.backend.floatx())*dx)[:,0]
        L = tf.concat([tf.ones((tf.shape(aspect_ratios)[0],1),aspect_ratios.dtype),aspect_ratios],-1)
        L = tf.einsum('ij,i->ij',L,Lx)
        npts_other_dimensions = tf.cast(tf.random.uniform((ndims-1,),dtype=tf.keras.backend.floatx())*tf.cast(random_output_shape_range[1:,1]-random_output_shape_range[1:,0],tf.keras.backend.floatx()),tf.int32) + random_output_shape_range[1:,0]
        dx_other_dimensions = tf.einsum('bd,d->bd',L[:,1:],tf.cast(1/(npts_other_dimensions-1),L.dtype))
        dx = tf.concat([dx,dx_other_dimensions],1)
    npts = tf.concat([tf.expand_dims(nx,0),npts_other_dimensions],0)
    maxpts = tf.reduce_max(random_output_shape_range,1)
    minpts = tf.reduce_min(random_output_shape_range,1)
    scaling_factor_for_oversized_dims = tf.reduce_max(tf.concat([[1.0],npts/maxpts],0))
    scaling_factor_for_undersized_dims = tf.reduce_min(tf.concat([[1.0],npts/minpts],0))
    scaling_factor = scaling_factor_for_oversized_dims if scaling_factor_for_oversized_dims > 1.0 else tf.reduce_max([scaling_factor_for_undersized_dims, tf.reduce_max(npts/maxpts)])
    #scaling_factor = tf.reduce_max([scaling_factor_for_oversized_dims,scaling_factor_for_undersized_dims])#more important to avoid oversized dims to prevent OOM errors
    npts = tf.cast(tf.cast(npts,tf.float64)/scaling_factor,tf.int32)
    return npts,dx
