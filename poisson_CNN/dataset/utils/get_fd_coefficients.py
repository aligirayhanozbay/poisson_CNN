import numpy as np
from scipy.special import factorial

def get_fd_coefficients(stencil_positions, order):
    '''
    Given a list of stencil positions (e.g. -3 -2 -1 0 1) and a derivative order, computes the finite difference coefficients associated with each stencil position

    Inputs:
    -stencil_positions: list of ints. contains the stencil positions
    -order: int. order of the derivative

    Output:
    -coefficients: np.array of floats. finite difference coefficients with delta_x = 1.0.
    '''
    stencil_positions = np.array(sorted(stencil_positions))
    stencil_coefficient_calculation_matrix = np.linalg.inv(np.array([stencil_positions**k for k in range(len(stencil_positions))]))
    stencil_order_vector = np.zeros((len(stencil_positions),))
    stencil_order_vector[order] = factorial(order)
    return np.einsum('ij,j->i',stencil_coefficient_calculation_matrix,stencil_order_vector)

if __name__=='__main__':
    stencil_positions = [-2,-1,0,1,2]
    order = 2
    print(get_fd_coefficients(stencil_positions,order))
    
