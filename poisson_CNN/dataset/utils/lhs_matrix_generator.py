import tensorflow as tf
from collections.abc import Iterable

def poisson_lhs_matrix(domain_shape, spacings = None, domain_includes_edges = True):

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
    
if __name__ == '__main__':
    import numpy as np
    ##unit test 1: random spacings, 3d domain, domain includes edges
    domain_shape = [100,50,75]
    spacings = [np.random.rand(k-1) for k in domain_shape]
    pm = poisson_lhs_matrix(domain_shape,spacings)
    print('---Unit Test 1: Random Spacings, 3d domain, domain includes edges---')
    if tf.reduce_sum(tf.keras.backend.shape(pm)[1:]-tf.cast(domain_shape,tf.int32)) == 0:
        print('Shape check passed')

    ##unit test 2: constant spacings, 3d domain, domain doesnt include edges, compare against pyamg
    import pyamg
    print('---Unit Test 2: Match pyamg---')
    domain_shape = [100,100,100]
    pm = poisson_lhs_matrix(domain_shape,domain_includes_edges=False)
    pm_pyamg = pyamg.gallery.poisson(domain_shape).data
    errors = []
    #import pdb
    #pdb.set_trace()
    for dim in range(pm.shape[0]):
        errors.append(tf.reduce_sum(tf.cast(tf.math.logical_not(tf.reshape(pm[dim],[-1]) == -pm_pyamg[dim]),tf.int32)))
    print('Total errors in pyamg comparison test (should be 0): ' + str(int(sum(errors))))
    
    ##unit test 3: 2nd derivative
    import scipy
    print('---Unit test 3: Compute 2nd derivate of x^2-2x+1 in the interval [0,2] sampled on 50 Chebyshev points---')
    npts = 50
    domain_shape = [npts-2]
    x,w = np.polynomial.chebyshev.chebgauss(npts)
    x = x+1
    y = x**2-2*x+1
    pm = scipy.sparse.dia_matrix((poisson_lhs_matrix(domain_shape,spacings = [x[1:]-x[:-1]], domain_includes_edges = False),[-1,0,1]),shape=(npts-2,npts-2)).toarray()
    grads = np.einsum('ij,j->i',pm,y[1:-1])
    grads[0] = grads[0]+y[0]*2/((x[1]-x[0])*(x[2]-x[0]))#adjust edge values in gradient computation
    grads[-1] = grads[-1]+y[-1]*2/((x[-1]-x[-2])*(x[-1]-x[-3]))
    print(grads-2)
    print('RMS error (reference value=2): ' + str(float(tf.reduce_mean((grads-2)**2)**0.5)))
    print('RMS error on inner pts (reference value=2): ' + str(float(tf.reduce_mean((grads[1:-1]-2)**2)**0.5)))

    ##unit test 4: 2nd derivative - regular mesh
    print('---Unit test 3: Compute 2nd derivate of x^2-2x+1 in the interval [0,2] sampled on 50 equispaced points---')
    npts = 50
    domain_shape = [npts-2]
    x = np.linspace(0,2,num=npts)
    y = x**2-2*x+1
    pm = scipy.sparse.dia_matrix((poisson_lhs_matrix(domain_shape, spacings = x[1]-x[0], domain_includes_edges = False), [-1,0,1]), shape=(npts-2,npts-2)).toarray()
    grads = np.einsum('ij,j->i',pm,y[1:-1])
    grads[0] = grads[0]+y[0]*2/((x[1]-x[0])*(x[2]-x[0]))
    grads[-1] = grads[-1]+y[-1]*2/((x[-1]-x[-2])*(x[-1]-x[-3]))
    print(grads-2)
    print('RMS error (reference value=2): ' + str(float(tf.reduce_mean((grads-2)**2)**0.5)))
    print('RMS error on inner pts (reference value=2): ' + str(float(tf.reduce_mean((grads[1:-1]-2)**2)**0.5)))
    
    
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

    
