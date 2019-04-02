import tensorflow as tf
import numpy as np
import itertools

def Lp_integral_norm(image_size, domain, n_quadpts = 10, quadpts_randomization = 0, p=2, mse_component_weight = 0.0):
    '''
    This function generates a function that takes 2 function(s) evaluated on a 2D grid, the dimensions of which are stored in image_size, on a rectangular domain and evaluates the Lp norm of their difference using Gauss-Legendre quadrature
    
    image_size              :     tuple with 2 elements.
    domain                  :     tuple with 4 elements containing (xmin,xmax,ymin,ymax)
    n_quadpts               :     no of GL quadrature points
    quadpts_randomization   :     if set to a nonzero value, the dataset inputted to the generated func. will be split into 2*quadpts_randomization+1 pieces and the integral for the ith piece will be computed with n_quadpts + i * quadpts_randomization GL pts (where i = [n_quadpts-quadpts_randomization, n_quadpts+quadpts_randomization])
    p                       :     order of the Lp norm
    
    The shape of the inputs to the generated function must be (batch_size,channels,image_size[0],image_size[1])
    
    User inputs to the closure (generated function):
    y_true                  :     labels
    y_pred                  :     images
    
    '''
    #match data types with keras
    if tf.keras.backend.floatx() == 'float64':
        dtype = tf.float64
    elif tf.keras.backend.floatx() == 'float32':
        dtype = tf.float32
    else:
        dtype = tf.float16
    
    #arrays to store values for different # of GL quad. pts. in accordance with quadpts_randomization. length of each: 2 * quadpts_randomization + 1
    interpolation_weights = [] #array that stores weights b_ii such that the interpolated value at x,y is b_ii*f(X_i, Y_j) within a rectangle bounded by [X_1,X_2] x [Y_1, Y_2] (shapes: (n_quadpts, n_quadpts, 4, 2))
    quadweights_list = [] #array that stores GL quad weights (shapes: (n_quadpts, n_quadpts))
    index_combinations_list = [] #array of tensors, where each tensor stores the indices for the 4 points between which every GL quad pt lies (shapes: (no of quadpts, no of quadpts, 4, 2))
    coords = np.array(np.meshgrid(np.linspace(domain[0], domain[1], image_size[0]),np.linspace(domain[2], domain[3], image_size[1]),indexing = 'xy'), dtype = np.float64).transpose((1,2,0)) #coordinates of each grid pt in the domain
    image_coords = [coords[0,:,0], coords[:,1,1]] #x and y coordinates separately
    c = np.array([np.array(0.5*(domain[1] - domain[0]),dtype=np.float64),np.array(0.5*(domain[3] - domain[2]),dtype=np.float64)]) #scaling coefficients - for handling domains other than [-1,1] x [-1,1]
    d = np.array([np.array(0.5*(domain[1] + domain[0]),dtype=np.float64),np.array(0.5*(domain[3] + domain[2]),dtype=np.float64)])
    for n in range(n_quadpts - quadpts_randomization, n_quadpts + quadpts_randomization+1): #loop over no of quadpts
        quadrature_x, quadrature_w = tuple([np.polynomial.legendre.leggauss(n)[i].astype(np.float64) for i in range(2)]) #quadrature weights and points
        quadpts = tf.constant(np.apply_along_axis(lambda x: x + d, 0, np.einsum('ijk,i->ijk',np.array(np.meshgrid(quadrature_x,quadrature_x,indexing = 'xy')),c)).transpose((1,2,0)),dtype = tf.float64)
        #quadweights = tf.reduce_prod(c)*tf.tensordot(tf.squeeze(quadrature_w),tf.squeeze(quadrature_w),axes = 0)
        quadweights = tf.reduce_prod(c) * tf.einsum('i,j->ij',tf.squeeze(quadrature_w),tf.squeeze(quadrature_w))
        indices = [[],[]] #indices between each quadrature point lies - indices[0] is in x-dir and indices[1] is in the y-dir
        quad_coords = [quadpts[0,:,0], quadpts[:,1,1]] #x and y coordinates of each quad pt respectively
        #find the indices of coords between which every quad. pt. lies
        for i in range(len(indices)):
            j=0
            #does not work if more than 2 quad pts fall within 1 cell - fix later
            while len(indices[i]) < quadpts.shape[0] and j<image_coords[i].shape[0]:
                try:
                    if abs(float(quad_coords[i][len(indices[i])] - image_coords[i][j])) == float(min(abs(quad_coords[i][len(indices[i])] - image_coords[i][j-1]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j+1]))):
                        if quad_coords[i][len(indices[i])] - image_coords[i][j] < 0:
                            indices[i].append((j-1,j))
                        else:
                            indices[i].append((j,j+1))
                except:
                    if abs(float(quad_coords[i][len(indices[i])] - image_coords[i][j])) == float(min(abs(quad_coords[i][len(indices[i])] - image_coords[i][j-1]), abs(quad_coords[i][len(indices[i])] - image_coords[i][j]))):
                        indices[i].append((j-1,j))
                j+=1
        
        index_combinations = np.zeros((quadpts.shape[0], quadpts.shape[1], 4 , 2), dtype = np.int32) #array storing the 4 index combinations on the original grid which surround each quad. pt.
        corners = np.zeros((quadpts.shape[0], quadpts.shape[1], 2 , 2), dtype = np.int32) #array storing the lower left corner and the upper right corner of each box stored in index_combinations. effectively this will contain [[xmin,ymin],[xmax,ymax]] for the rectangle around each quad pt.
        s=np.array(indices)
        for i in range(n):
            for j in range(n):
                index_combinations[i,j,:,:] = np.array(list(itertools.product(np.array(s)[0,i,:],np.array(s)[1,j,:])))
        for i in range(n):
            for j in range(n):
                corners[i,j,:,:] = np.array([s[0,i,:],s[1,j,:]])
        corners = corners.transpose((0,1,3,2))
        corner_coords = tf.gather_nd(tf.transpose(coords,(1,0,2)),corners)
        
        #compute the coefficients [b_11,b_12,b_21,b_22]
        #steps:
        #1. compute transpose(invert(array([[1,xmin,ymin,xmin*ymin],[1,xmin,ymax,xmin*ymax],[1,xmax,ymin,xmax*ymin],[1,xmax,ymax,xmax*ymax]]))) for the rectangle around each quad pt.
        #2. compute array([1,x_quadpt, y_quadpt, x_quadpt*y_quadpt]) for each quadpt
        #3. multiply the result of 1 and 2 for each quad pt.
        interpolation_matrix = np.ones((n,n,4,4))
        interpolation_matrix[:,:,0:2,1] = np.einsum('ijk,ij->ijk',interpolation_matrix[:,:,0:2,1],corner_coords[:,:,0,0])
        interpolation_matrix[:,:,2:,1] = np.einsum('ijk,ij->ijk',interpolation_matrix[:,:,2:,1],corner_coords[:,:,1,0])
        interpolation_matrix[:,:,(0,2),2] = np.einsum('ijk,ij->ijk',interpolation_matrix[:,:,(0,2),2],corner_coords[:,:,0,1])
        interpolation_matrix[:,:,(1,3),2] = np.einsum('ijk,ij->ijk',interpolation_matrix[:,:,(1,3),2], corner_coords[:,:,1,1])
        interpolation_matrix[:,:,:,3] *= np.multiply(interpolation_matrix[:,:,:,1], interpolation_matrix[:,:,:,2])
        interpolation_matrix = tf.transpose(tf.linalg.inv(interpolation_matrix), (0,1,3,2))
        q = np.ones((n,n,4))
        q[:,:,1] = tf.transpose(quadpts[:,:,0])
        q[:,:,2] = tf.transpose(quadpts[:,:,1])
        q[:,:,3] = np.multiply(q[:,:,1],q[:,:,2])
        
        b = tf.einsum('ijkl, ijl->ijk', tf.constant(interpolation_matrix), tf.constant(q))
        
        #store results for the closure
        quadweights_list.append(tf.cast(quadweights,dtype))
        index_combinations_list.append(index_combinations)
        interpolation_weights.append(tf.cast(b, dtype))

    if int(tf.__version__[0]) < 2:
        @tf.contrib.eager.defun
        def Lp_integrate_batch(inp):
            '''
            Helper function to facilitate quad pt randomization

            Given the bilinear interp. weights b, GL quad. weights w, Lp norm order p and the indices bounding each quad pt ind, computes the Lp norm for each channel/batch element
            '''
            #unpack values
            data = tf.transpose(inp[0], (2,3,1,0))
            b = inp[1]
            w = inp[2]
            ind = inp[3]
            p = inp[4]

            #get the points from the image to perform interpolation on
            interp_pts = tf.squeeze(tf.gather_nd(data, ind))

            #multiply image values with the weights b ti interpolate original image onto the GL quadrature points
            if data.shape[-1] == 1: #needed to handle if batch dimension is 1
                interp_pts = tf.expand_dims(interp_pts, axis = 3)
            values_at_quad_pts = tf.einsum('ijkl, ijk->ijl', interp_pts, b)

            #compute Lp norm
            return tf.pow(tf.reduce_sum(tf.einsum('ij,ijk->ijk',w,tf.pow(values_at_quad_pts, p)), axis = (0,1)), 1/p)

        @tf.contrib.eager.defun
        def Lp_integrate(y_true,y_pred, b=interpolation_weights, w=quadweights_list, ind=index_combinations_list, p=p, mse_component_weight = mse_component_weight):
            '''
            Split the batch, pack with the appropriate parameters and obtain the integrals
            '''
            if mse_component_weight == 0.0:
                return tf.reduce_mean(tf.concat(list(map(Lp_integrate_batch, zip(tf.split(y_true-y_pred, len(b)), itertools.cycle(b), itertools.cycle(w), itertools.cycle(ind), itertools.repeat(p)))), 0))
            else:
                return tf.reduce_mean(mse_component_weight * tf.keras.losses.mean_squared_error(y_true,y_pred)) + tf.reduce_mean(tf.concat(list(map(Lp_integrate_batch, zip(tf.split(y_true-y_pred, len(b)), itertools.cycle(b), itertools.cycle(w), itertools.cycle(ind), itertools.repeat(p)))), 0))
    
    else:
        @tf.function
        def Lp_integrate_batch(inp):
            '''
            Helper function to facilitate quad pt randomization

            Given the bilinear interp. weights b, GL quad. weights w, Lp norm order p and the indices bounding each quad pt ind, computes the Lp norm for each channel/batch element
            '''
            #unpack values
            data = tf.transpose(inp[0], (2,3,1,0))
            b = inp[1]
            w = inp[2]
            ind = inp[3]
            p = inp[4]
            
            #get the points from the image to perform interpolation on
            interp_pts = tf.squeeze(tf.gather_nd(data, ind))

            if data.shape[-1] == 1: #needed to handle if batch dimension is 1
                interp_pts = tf.expand_dims(interp_pts, axis = 3)
                
            #multiply image values with the weights b ti interpolate original image onto the GL quadrature points
            values_at_quad_pts = tf.einsum('ijkl, ijk->ijl', interp_pts, b)

            #compute Lp norm
            return tf.pow(tf.reduce_sum(tf.einsum('ij,ijk->ijk',w,tf.pow(values_at_quad_pts, p)), axis = (0,1)), 1/p)

        @tf.function
        def Lp_integrate(y_true,y_pred, b=interpolation_weights, w=quadweights_list, ind=index_combinations_list, p=p, mse_component_weight = mse_component_weight):
            '''
            Split the batch, pack with the appropriate parameters and obtain the integrals
            '''
            if mse_component_weight == 0.0:
                return tf.reduce_mean(tf.concat(list(map(Lp_integrate_batch, zip(tf.split(y_true-y_pred, len(b)), itertools.cycle(b), itertools.cycle(w), itertools.cycle(ind), itertools.repeat(p)))), 0))
            else:
                return tf.reduce_mean(mse_component_weight * tf.keras.losses.mean_squared_error(y_true,y_pred)) + tf.reduce_mean(tf.concat(list(map(Lp_integrate_batch, zip(tf.split(y_true-y_pred, len(b)), itertools.cycle(b), itertools.cycle(w), itertools.cycle(ind), itertools.repeat(p)))), 0))
    return Lp_integrate
    
