import tensorflow as tf

@tf.function
def integrate_piecewise_sigmoid(L1,L2):
    '''
    Evaluates the definite integral of the piecewise "sigmoid" function s(x,a,b) = a if x < a, x if a<x<b, b if x>b
    
    Inputs:
    -L1: float tensor of shape (batch_size,2). Integration bounds. L1[...,0] are the lower integration bounds and L1[...,1] are the upper bounds.
    -L2: float tensor of shape (batch_size,2). Contains the parameters (a,b) used in the description above.

    Outputs:
    Float tensor of shape (batch_size,) containing the integration result
    '''
    pre_first_threshold_result = tf.reduce_max(tf.stack([(L2[...,0] - L1[...,0]) * L2[...,0], tf.zeros(tf.shape(L1)[:-1], L1.dtype)],-1),-1)
    post_second_threshold_result = tf.reduce_max(tf.stack([(L1[...,1] - L2[...,1]) * L2[...,1], tf.zeros(tf.shape(L1)[:-1], L1.dtype)],-1),-1)
    
    inter_threshold_result_lower_integration_domain_boundary = tf.reduce_max(tf.stack([L2[...,0],L1[...,0]],-1),-1)
    inter_threshold_result_upper_integration_domain_boundary = tf.reduce_min(tf.stack([L2[...,1],L1[...,1]],-1),-1)
    inter_threshold_result = 0.5*(inter_threshold_result_upper_integration_domain_boundary**2 - inter_threshold_result_lower_integration_domain_boundary**2)
    
    return pre_first_threshold_result + post_second_threshold_result + inter_threshold_result

@tf.function
def compute_proportion_of_AR_range_under_1(L1,L2):
    '''
    Given two sets of domain sizes, computes the proportion of possible aspect ratio values - defined as (domain size in range L1)/(domain size in range L2) - that fall under 1.0

    Inputs:
    -L1: float tensor of shape (batch_size,2) or (2,) or (1,2). Range of possible values for the 1st dimension.
    -L2: float tensor of shape (batch_size,2). Range of possible values for the 2nd dimension.

    Outputs:
    Float tensor of shape (batch_size,)
    '''
    if tf.shape(tf.shape(L1))[0] == 1:
        L1 = tf.expand_dims(L1,0)
    if tf.shape(L1)[0] == 1:
        L1 = tf.tile(L1,[tf.shape(L2)[0],1])
    return ((L2[...,1])*(L1[...,1]-L1[...,0]) - integrate_piecewise_sigmoid(L1,L2))/((L2[...,1]-L2[...,0])*(L1[...,1]-L1[...,0]))

@tf.function
def compute_domain_size_range(output_shape_range, dx_range):
    '''
    Given a range of possible output domain shapes and grid spacing ranges, outputs the possible resulting domain sizes.

    Inputs:
    -output_shape_range: int tensor of shape [ndims,2]. Ranges of possible domain shapes.
    -dx_range: float tensor of shape [ndims,2]. Ranges of possible grid spacings.

    Outputs:
    Float tensor of shape [ndims,2]
    '''
    Lmax = tf.cast(output_shape_range[...,1]-1,dx_range.dtype)*dx_range[...,1]
    Lmin = tf.cast(output_shape_range[...,0]-1,dx_range.dtype)*dx_range[...,0]
    return tf.stack([Lmin,Lmax],-1)

@tf.function
def generate_uniformly_distributed_aspect_ratios(output_shape_range, dx_range = None, samples = 1):
    '''
    Given a range of output shapes and, optionally, grid spacing ranges, generates uniformly distributed aspect ratios for grids with such parameters.

    Inputs:
    -output_shape_range: int tensor of shape [ndims,2]. Ranges of possible domain shapes.
    -dx_range: None or float tensor of shape [ndims,2]. Ranges of possible grid spacings. If left as None, identical grid spacings across all dimensions will be assumed.
    -samples: int. Determines how many samples to generate.

    Outputs:
    Float tensor of shape [samples, ndims-1]
    '''
    output_shape_range = tf.convert_to_tensor(output_shape_range)
    if dx_range is None:
        domain_size_range = tf.cast(output_shape_range,tf.float32)-1.0
    else:
        domain_size_range = compute_domain_size_range(output_shape_range, dx_range)
    max_ar = domain_size_range[0,1]/domain_size_range[1:,0]
    min_ar = domain_size_range[0,0]/domain_size_range[1:,1]
    proportions_of_AR_ranges_under_1 = compute_proportion_of_AR_range_under_1(domain_size_range[0],domain_size_range[1:])

    ar_values_under_1_mask = tf.cast(tf.transpose(tf.map_fn(lambda x: x[0] < x[1] , (tf.random.uniform((tf.shape(proportions_of_AR_ranges_under_1)[0],samples)), proportions_of_AR_ranges_under_1), dtype = tf.bool),[1,0]), domain_size_range.dtype)
    ar_values_under_1_upper_bounds = tf.reduce_min(tf.stack([max_ar,tf.ones(tf.shape(max_ar),dtype=max_ar.dtype)]))
    ar_values_above_1_lower_bounds = tf.reduce_max(tf.stack([min_ar,tf.ones(tf.shape(min_ar),dtype=min_ar.dtype)]))
    
    ar_values = ar_values_under_1_mask * ((ar_values_under_1_upper_bounds-min_ar)*tf.random.uniform(tf.shape(ar_values_under_1_mask),dtype=domain_size_range.dtype) + min_ar) + (1-ar_values_under_1_mask) * ((max_ar - ar_values_above_1_lower_bounds)*tf.random.uniform(tf.shape(ar_values_under_1_mask),dtype=domain_size_range.dtype) + ar_values_above_1_lower_bounds)
    return ar_values
