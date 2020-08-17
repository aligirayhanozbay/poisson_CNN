import tensorflow as tf

@tf.function
def flip_and_rotate_tensor(inp, rotation_axis = 4, rotation_count = 0, flip_axes = None, data_format = 'channels_first'):
    '''
    Applies a number of 90-deg rotations around a given axis and/or flips the spatial/data dimensions of a tensor.

    All rotations are applied through reflection (tf.transpose) + axis flipping
    
    Inputs:
    -inp: Tensor of shape [batch_size, n_channels, spatial_dim_1,..., spatial_dim_n] where n<=3. Contains the data.
    -rotation_axis: int between 2 and 4. Tensor dimension around which to rotate. Necessary only when rotation_count > 0 and n == 3. Supplying this otherwise may cause issues.
    -rotation_count: int. No 90 deg rotations around the specified rotation axis.
    -flip_axes: int tensor. Axes to flip with tf.reverse. Note that this flip is performed IN ADDITION TO any flips that may be required to 'rotate' the tensor following tf.transpose.
    -data_format: str. Same as in tf.keras.
    '''
    n_tensor_dims = tf.rank(inp)
    inp = tf.reshape(inp, tf.concat([tf.shape(inp), tf.ones((5-n_tensor_dims,), dtype = tf.int32)],0))
    #output_squeeze_mask = tf.concat([tf.zeros(tf.shape(tf.shape(inp)), dtype = tf.bool), tf.ones((5-n_tensor_dims,), dtype = tf.bool)],0)
    n_spatial_dims = 3#n_tensor_dims-2
    spatial_dims_start = 2 if data_format == 'channels_first' else 1
    spatial_dims_end = spatial_dims_start + n_spatial_dims
    rotation_axis_spatial_dim_index = rotation_axis - spatial_dims_start
    flip_count_by_axis = tf.reduce_sum(tf.one_hot([] if flip_axes is None else flip_axes, n_spatial_dims+2, dtype = tf.int32),0)
    if rotation_count != 0:
        spatial_dim_indices = tf.range(spatial_dims_start, spatial_dims_end)
        dimension_indices_before_rotation_axis = spatial_dim_indices[:rotation_axis_spatial_dim_index]
        dimension_indices_after_rotation_axis = spatial_dim_indices[rotation_axis_spatial_dim_index+1:]
        dimension_indices_to_rotate = tf.concat([dimension_indices_before_rotation_axis,dimension_indices_after_rotation_axis],0)
        axis_rotation_count = rotation_count % (n_spatial_dims - 1)
        rotated_dimension_indices = tf.concat([dimension_indices_to_rotate[axis_rotation_count:], dimension_indices_to_rotate[:axis_rotation_count]],0)
        rotated_spatial_dim_indices = tf.concat([rotated_dimension_indices[:rotation_axis_spatial_dim_index], tf.expand_dims(spatial_dim_indices[rotation_axis_spatial_dim_index],0), rotated_dimension_indices[rotation_axis_spatial_dim_index:]],0)
        transpose_indices = tf.concat([[0,1],rotated_spatial_dim_indices],0) if data_format == 'channels_first' else tf.concat([[0],rotated_spatial_dim_indices,[1]],0)
        out = tf.transpose(inp, transpose_indices)
        rotation_flip_requirement_lookup_table = tf.constant([[0,0],[1,0],[1,1],[0,1]], dtype = tf.int32)
        rotation_flip_requirement = rotation_flip_requirement_lookup_table[(tf.abs(rotation_count) % 4) * tf.math.sign(rotation_count)]
        flip_count_by_axis = tf.tensor_scatter_nd_add(flip_count_by_axis, tf.expand_dims(dimension_indices_to_rotate,-1), rotation_flip_requirement)
    else:
        out = inp
    flip_count_by_axis = flip_count_by_axis % 2
    out = tf.reverse(out, tf.where(flip_count_by_axis == 1)[:,0])
    batch_and_channel_dim_shapes = tf.stack([tf.shape(inp)[0], tf.shape(inp)[1 if data_format == 'channels_first' else -1]],0)
    out = tf.reshape(out, tf.concat([tf.expand_dims(tf.reduce_prod(batch_and_channel_dim_shapes),0), tf.shape(out)[slice(2,None) if data_format == 'channels_first' else slice(1,-1)]],0))
    out = tf.map_fn(tf.squeeze, out)
    final_shape = tf.concat([tf.expand_dims(batch_and_channel_dim_shapes[0],0), tf.expand_dims(batch_and_channel_dim_shapes[1],0), tf.shape(out)[1:]],0) if data_format == 'channels_first' else tf.concat([tf.expand_dims(batch_and_channel_dim_shapes[0],0), tf.shape(out)[1:], tf.expand_dims(batch_and_channel_dim_shapes[1],0)],0)
    out = tf.reshape(out, final_shape)
    return out

if __name__ == '__main__':
    #s = tf.random.uniform((10,1,10,11,12,13))
    nx = 5
    ny = 4
    s = tf.range(nx)
    q = tf.range(ny)
    r = tf.expand_dims(s,1) + tf.expand_dims(q,0)
    r2 = tf.expand_dims(r,-1)
    r3 = tf.reshape(r,[1,1,nx,ny])
    rot_count = -1
    print(r)
    print(tf.image.rot90(r2,rot_count)[...,0])
    print(flip_and_rotate_tensor(r3, rotation_count = rot_count, flip_axes = [2], data_format = 'channels_first'))
        
            
