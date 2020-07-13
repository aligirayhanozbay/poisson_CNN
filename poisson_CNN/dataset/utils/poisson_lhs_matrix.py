import tensorflow as tf
from collections.abc import Iterable
from .assign_to_tensor_index import assign_to_tensor_index

@tf.function
def tile_tensor_to_shape(data, shape, axes = None):
    #map dimensions of target shape and original shape to each other if not already done so using the optional parameter axes
    data_shape = tf.keras.backend.shape(data)
    ndims_data = tf.keras.backend.shape(data_shape)[0]
    ndims_target = tf.keras.backend.shape(shape)[0]
        
    if axes is None:
        matching_dims = tf.cast(tf.where(tf.abs(shape-tf.expand_dims(data_shape,1))==0),tf.int32)#first creates ndims x len(shape) tensor showing in each row where the data_shape matches target shape. then, tf.where converts to an 2 x (total no of matches) tensor giving coords of each match
        rows_with_matches, row_ids = tf.unique(matching_dims[:,0])
        _, unique_shape_ids = tf.unique(data_shape)
        encountered_dim_sizes = -tf.ones((tf.reduce_max(unique_shape_ids)+1,),dtype=tf.int32)
        dim_mapping = tf.zeros((ndims_data,),dtype=tf.int32)#list keeping track of to which index in shape each data dim is mapped to
        for dim in range(ndims_data):
            dimr = unique_shape_ids[dim]#data_shape[dim].experimental_ref()
            encountered_dim_sizes = assign_to_tensor_index(encountered_dim_sizes,encountered_dim_sizes[dimr]+1,dimr)
            dim_mapping = assign_to_tensor_index(dim_mapping, matching_dims[tf.where(matching_dims[:,0]==rows_with_matches[dim])[0,0]:tf.where(matching_dims[:,0]==rows_with_matches[dim])[-1,0]+1][encountered_dim_sizes[dimr],1],dim)#for each dim size, find the mapping onto shape. the index of the mapping is equal to (encountered_dim_sizes[dim])th time an axis size has been encountered in shape.
    else:
        dim_mapping = axes

    expanded_shape = tf.ones((ndims_target,),dtype=tf.int32)#expand the input tensor with additional 1-size dimensions to prepare for tiling
    for dim in range(ndims_data):
        expanded_shape = assign_to_tensor_index(expanded_shape, data_shape[dim], dim_mapping[dim])
    data = tf.reshape(data,expanded_shape)

    tiling_list = tf.convert_to_tensor(shape)
    for dim in range(ndims_data):
        tiling_list = assign_to_tensor_index(tiling_list, 1, dim_mapping[dim])
    data = tf.tile(data,tiling_list)#tile and return
    return data

@tf.function
def place_diagonal(data, stencil_coordinate, coefficient_tensor_slice):
    domain_shape = tf.keras.backend.shape(coefficient_tensor_slice)
    ndims = tf.keras.backend.shape(domain_shape)[0]
    stencil_coordinate = tf.convert_to_tensor(stencil_coordinate)
    #sl = [slice(tf.reduce_max([stencil_coordinate[k],0]),tf.reduce_min([stencil_coordinate[k],0])+domain_shape[k]) for k in range(ndims)]
    sl = []
    for k in range(ndims):
        sl.append(slice(tf.reduce_max([stencil_coordinate[k],0]),tf.reduce_min([stencil_coordinate[k],0])+domain_shape[k]))
    print(domain_shape-tf.abs(stencil_coordinate))
    #data = tf.reshape(data, 
    tiled_data = tile_tensor_to_shape(data, domain_shape-tf.abs(stencil_coordinate))
    print('asdasdasd')
    coefficient_tensor_slice[sl].assign(tiled_data)
    return
    
    

def place_finite_difference_coefficients(data, stencil_coordinate_mapping, domain_shape = None, coefficients_tensor = None):
    stencil_coordinate_mapping = tf.convert_to_tensor(stencil_coordinate_mapping)
    ndims = stencil_coordinate_mapping.shape[0]
    if domain_shape is None:#automatically infer domain shape and build the output variable if not provided
        domain_shape = tf.convert_to_tensor((data[k].shape for k in range(ndims)))
    if coefficients_tensor is None:
        coefficients_tensor = tf.Variable(tf.zeros(domain_shape,tf.keras.backend.floatx()))
    
    
    return

def _variablemesh_compute_finite_difference_coefficients(domain_shape, spacings = None, domain_includes_edges = True):

    ndims = len(domain_shape)

    domain_shape = tf.cast(domain_shape,tf.int32)
    if not domain_includes_edges:
        domain_shape = domain_shape+2
    
    if spacings is None:
        spacings = [tf.ones((domain_shape[dim]-1,), dtype = tf.keras.backend.floatx()) for dim in range(ndims)]
    elif isinstance(spacings, float):
        spacings = [spacings*tf.ones((domain_shape[dim]-1,), dtype = tf.keras.backend.floatx()) for dim in range(ndims)]
    elif isinstance(spacings, Iterable) and isinstance(spacings[0],float):
        spacings = [spacings[dim]*tf.ones((domain_shape[dim]-1,), dtype = tf.keras.backend.floatx()) for dim in range(ndims)]
        
    variable_spacing_coefficients = []
    for k in range(ndims):
        spacing = tf.cast(spacings[k],tf.keras.backend.floatx())
        variable_spacing_coefficients.append(2/(spacing[1:]*spacing[:-1]*(spacing[1:]+spacing[:-1])))
        
    return variable_spacing_coefficients

def _finitedifference_poisson_matrix(domain_shape, spacings = None, domain_includes_edges = True):

    ndims = len(domain_shape)

    domain_shape = tf.cast(domain_shape,tf.int32)
    if not domain_includes_edges:
        domain_shape = domain_shape+2
    
    if spacings is None:
        spacings = [tf.ones((domain_shape[dim]-1,), dtype = tf.keras.backend.floatx()) for dim in range(ndims)]
    elif isinstance(spacings, float):
        spacings = [spacings*tf.ones((domain_shape[dim]-1,), dtype = tf.keras.backend.floatx()) for dim in range(ndims)]
    elif isinstance(spacings, Iterable) and isinstance(spacings[0],float):
        spacings = [spacings[dim]*tf.ones((domain_shape[dim]-1,), dtype = tf.keras.backend.floatx()) for dim in range(ndims)]
        
    variable_spacing_coefficients = []
    for k in range(ndims):
        spacing = tf.cast(spacings[k],tf.keras.backend.floatx())
        variable_spacing_coefficients.append(2/(spacing[1:]*spacing[:-1]*(spacing[1:]+spacing[:-1])))
            
    #use the fact that data across matrix diagonals in scipy.dia_matrix object is accessible via LHS.data to modify the LHS matrix
    #build the Poisson matrix diagonals. values are first modified in the n dimensional representation as opposed to flattened for ease of use.
    coeffs = tf.Variable(tf.keras.backend.zeros([2*ndims+1] + list(domain_shape)),dtype=tf.keras.backend.floatx())
    
    for dim in range(ndims):#place coefficients into the array for subdiagonals
        sl = tuple([dim] + [slice(0,domain_shape[k]) for k in range(dim)] + [slice(0,domain_shape[dim]-2)] + [Ellipsis])#slice for accessing the sub-array; for subdiagonals the coefficients corresponding to A[...,-1,...] are left as 0
        tiling_list = tf.concat([domain_shape[:dim],[1],domain_shape[dim+1:]],axis=0)
        coeff = variable_spacing_coefficients[dim]*spacings[dim][1:]#compute matrix entries
        coeff = tf.reshape(coeff, [1 for k in range(dim)] + [tf.shape(coeff)[0]] + [1 for k in range(dim+1,ndims)])#reshape to add extra 1-length dimensions to prepare for tiling
        coeffs[sl] .assign( tf.tile(coeff, tiling_list) )#tile to match shape and put in main array

    coeff = 0#place coefficients into the array for the main diagonal
    for dim in range(ndims):
        coeff = coeff - tf.reshape(variable_spacing_coefficients[dim]*(spacings[dim][:-1]+spacings[dim][1:]), [1 for j in range(dim)] + [variable_spacing_coefficients[dim].shape[0]] + [1 for j in range(dim+1,ndims)])
    coeffs[tuple([ndims] + [slice(1,size-1) for size in domain_shape])] .assign(coeff)

    #place coefficients into the array for the superdiagonals
    for dim in range(ndims):
        diagonal_idx = 2*ndims-dim
        sl = tuple([diagonal_idx] + [slice(0,domain_shape[k]) for k in range(dim)] + [slice(2,domain_shape[dim])] + [Ellipsis])
        tiling_list = tf.concat([domain_shape[:dim],[1],domain_shape[dim+1:]],axis=0)
        coeff = variable_spacing_coefficients[dim]*spacings[dim][:-1]
        coeff = tf.reshape(coeff, [1 for k in range(dim)] + [tf.shape(coeff)[0]] + [1 for k in range(dim+1,ndims)])
        coeffs[sl] .assign( tf.tile(coeff, tiling_list) )

    if domain_includes_edges:
        return coeffs
    else:
        return coeffs[tuple([Ellipsis] + [slice(1,domain_shape[k]-1) for k in range(ndims)])]

def _stretchingfuction_compute_finite_difference_coefficients():
    return
    
def poisson_lhs_matrix(domain_shape, spacings_or_stretching_function = None, domain_includes_edges = True):

    if not callable(spacings_or_stretching_function):#if spacings_or_stretching_function is not a stretching function, treat as though a regular finite difference problem
        return _finitedifference_poisson_matrix(domain_shape, spacings_or_stretching_function, domain_includes_edges)

    ndims = len(domain_shape)
    h = spacings_or_stretching_function
    s=[tf.linspace(0.0,1.0,domain_shape[k]) for k in range(ndims)]
    y=[]
    grads_1 = []
    grads_2 = []
    for k in range(ndims):
        with tf.GradientTape() as tape_d2:
            tape_d2.watch(s[k])
            with tf.GradientTape() as tape_d1:
                tape_d1.watch(s[k])
                y.append(h(s[k]))
            grads_1.append(tape_d1.gradient(y[k],s[k]))
        grads_2.append(tape_d2.gradient(grads_1[k],s[k]))
    print(grads_2)
    return
    
if __name__ == '__main__':
    import numpy as np
    ##unit test: test tile_tensor_to_shape
    q = tf.Variable(tf.zeros((4,6,4,5,4)))
    q[0,...].assign(q[0,...]+1)
    q = tf.convert_to_tensor(q)
    s = [3,4,5,6,7,4,4]
    print('---Unit Test: Tile tensor to shape---')
    a = tile_tensor_to_shape(q,s)
    if tf.reduce_all(a.shape == tf.convert_to_tensor(s)):
        print('Shape check passed')
    else:
        print('Incorrect dimension sizes in shape check')
        print('Target shape: ' + str(s) + ' | Encountered shape: ' + str(list(a.shape)))
    if tf.reduce_sum(tf.abs(a[0,1,...]))==0.0:
        print('Correct shape retention check 1 passed')
    else:
        print('Incorrect values encountered in correct shape retention check 1. Sum of sub-array (target 0.0): ' + str(float(tf.reduce_sum(tf.abs(a[0,0,...])))))
    if tf.reduce_sum(tf.abs(a[0,0,...]-1.0))==0.0:
        print('Correct shape retention check 2 passed')
    else:
        print('Incorrect values encountered in correct shape retention check 2. Sum of sub-array (target 0.0): ' + str(float(tf.reduce_sum(tf.abs(a[0,0,...])))))

    ##unit test: place_diagonal with a flattened input, stencil size of 5 and no domain_shape or ndims provided
    print('---Unit Test: Place diagonal---')
    q = tf.random.uniform((3,5))
    s = tf.Variable(tf.zeros((3,5,11,9)),dtype=q.dtype)
    place_diagonal(q,(0,0,-2,0),s)
    
    

    
    ##unit test : random spacings, 3d domain, domain includes edges
    domain_shape = [100,50,75]
    spacings = [np.random.rand(k-1) for k in domain_shape]
    pm = poisson_lhs_matrix(domain_shape,spacings)
    print('---Unit Test: Random Spacings, 3d domain, domain includes edges---')
    if tf.reduce_sum(tf.keras.backend.shape(pm)[1:]-tf.cast(domain_shape,tf.int32)) == 0:
        print('Shape check passed')

    ##unit test: constant spacings, 3d domain, domain doesnt include edges, compare against pyamg
    import pyamg
    print('---Unit Test: Match pyamg---')
    domain_shape = [100,100,100]
    pm = poisson_lhs_matrix(domain_shape,domain_includes_edges=False)
    pm_pyamg = pyamg.gallery.poisson(domain_shape).data
    errors = []
    #import pdb
    #pdb.set_trace()
    for dim in range(pm.shape[0]):
        errors.append(tf.reduce_sum(tf.cast(tf.math.logical_not(tf.reshape(pm[dim],[-1]) == -pm_pyamg[dim]),tf.int32)))
    print('Total errors in pyamg comparison test (should be 0): ' + str(int(sum(errors))))
    
    ##unit test: 2nd derivative on differently spaced mesh. numerics mean that with a chebyshev pt distribution the error increases as no of pts increase!
    import scipy
    print('---Unit test: Compute 2nd derivate of x^2-2x+1 in the interval [0,2] sampled on 50 Chebyshev points---')
    npts = 100
    domain_shape = [npts-2]
    x,w = np.polynomial.chebyshev.chebgauss(npts)
    x = x+1
    y = x**2-2*x+1
    pm = scipy.sparse.dia_matrix((poisson_lhs_matrix(domain_shape, [x[1:]-x[:-1]], domain_includes_edges = False),[-1,0,1]),shape=(npts-2,npts-2)).toarray()
    grads = np.einsum('ij,j->i',pm,y[1:-1])
    grads[0] = grads[0]+y[0]*2/((x[1]-x[0])*(x[2]-x[0]))#adjust edge values in gradient computation
    grads[-1] = grads[-1]+y[-1]*2/((x[-1]-x[-2])*(x[-1]-x[-3]))
    print('Pointwise error values: ')
    print(grads-2)
    print('RMS error (reference value=2): ' + str(float(tf.reduce_mean((grads-2)**2)**0.5)))
    print('RMS error on inner pts (reference value=2): ' + str(float(tf.reduce_mean((grads[1:-1]-2)**2)**0.5)))
    print('Maximal adjacent spacing discrepancy :' + str(float(tf.reduce_max(tf.abs(x[1:]-x[:-1])))))

    ##unit test 4: 2nd derivative - regular mesh
    print('---Unit test: Compute 2nd derivate of x^2-2x+1 in the interval [0,2] sampled on 50 equispaced points---')
    npts = 100
    domain_shape = [npts-2]
    x = np.linspace(0,2,num=npts)
    y = x**2-2*x+1
    pm = scipy.sparse.dia_matrix((poisson_lhs_matrix(domain_shape, x[1]-x[0], domain_includes_edges = False), [-1,0,1]), shape=(npts-2,npts-2)).toarray()
    grads = np.einsum('ij,j->i',pm,y[1:-1])
    grads[0] = grads[0]+y[0]*2/((x[1]-x[0])*(x[2]-x[0]))
    grads[-1] = grads[-1]+y[-1]*2/((x[-1]-x[-2])*(x[-1]-x[-3]))
    print('Pointwise error values: ')
    print(grads-2)
    print('RMS error (reference value=2): ' + str(float(tf.reduce_mean((grads-2)**2)**0.5)))
    print('RMS error on inner pts (reference value=2): ' + str(float(tf.reduce_mean((grads[1:-1]-2)**2)**0.5)))
    print('Maximal adjacent spacing discrepancy :' + str(float(tf.reduce_max(tf.abs(x[1:]-x[:-1])))))

    
    ##unit test 5: stretching function
    h = lambda s:s**2
    domain_shape = [20,40]
    poisson_lhs_matrix(domain_shape,h)
    

    
    ##tf.constant([100,50,75])#
    # domain_shape = [10,10,10]
    # #
    # dx = 0.005
    # spacings = [dx*np.ones((z-1,)) for z in domain_shape]
    
    # import pyamg
    # q = pyamg.gallery.poisson(domain_shape)
    # for dim in range(q.data.shape[0]):
    #     print(tf.reduce_sum(tf.cast(tf.reshape(s[dim],[-1]) == -q.data[dim]/(dx**2),tf.int32)) - s[dim].shape[0])
    # import pdb
    # pdb.set_trace()
    #print([k.shape for k in spacings])
    #print(poisson_lhs_matrix(domain_shape,spacings)[3])

    
