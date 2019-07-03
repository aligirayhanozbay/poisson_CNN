#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 13:36:21 2019

@author: ali
"""
import numpy as np
import scipy
#from inspect import signature
import tensorflow as tf
#import pdb

class Boundary1D:
    '''
        A class that implements a 1D curved boundary defined via piecewise interpolation of provided coordinates
        
        Init parameters:
            boundary_type               : 'Dirichlet' (u = g on the boundary), 'von Neumann' (du/dn = g on the boundary) or 'Robin' (du/dn + a*u = g on the boundary)
            coordinates                 : a list or numpy array containing the coordinates of the boundary supplied in order
            error_type                  : Currently Lp spaces are supported
            RHS_function                : g. Supply a function taking a single argument. If it's a parametric function, set boundary_rhs_is_parametric to True. If not, (i.e. if the function takes x and y coords) it should  still take a single argument but the supplied argument should contain x = arg[0] and y = arg[1]
            robin_bc_alpha              : value of a used in the Robin BC
            interpolation_order         : order of spline interpolation to be used between supplied coords.
            orientation                 : clockwise boundaries will rotate normal vector 90 degrees CCW of the tangent and vice versa
            boundary_rhs_is_parametric  : determines if a parametric function g(t) or a 'regular' function g(x,y) is specified as the rhs of the boundary
    '''
    
    def __init__(self, boundary_type, coordinates, error_type = 'L2', RHS_function = lambda x: 0, label = None, robin_bc_alpha = None, interpolation_order = 1, orientation = 'counterclockwise', derivative_regularizer_coefficient = 0.0, boundary_rhs_is_parametric = False, n_threads = 8, dtype = 'float64'):
        
        self.dtype = dtype
        self.error_type = error_type
        self.orientation = orientation
        self.boundary_type = boundary_type
        self.coordinates = np.array(coordinates)
        self.label = label
        self.RHS_function = RHS_function
        if robin_bc_alpha:
            self.robin_bc_alpha = robin_bc_alpha
        
        self.t = self.get_arc_length(self.coordinates)
        self.x_interpolant = scipy.interpolate.InterpolatedUnivariateSpline(self.t,self.coordinates[:,0], k = interpolation_order)
        self.y_interpolant = scipy.interpolate.InterpolatedUnivariateSpline(self.t,self.coordinates[:,1], k = interpolation_order)
        self.dxdt = self.x_interpolant.derivative(n = 1)
        self.dydt = self.y_interpolant.derivative(n = 1)
        self.derivative_regularizer_coefficient = derivative_regularizer_coefficient
        self.boundary_rhs_is_parametric = boundary_rhs_is_parametric        
        self.n_threads = 8
        
        if self.orientation == 'counterclockwise':
            self.tangent_to_normal_rotation_matrix = np.array([[0,1],[-1,0]])
        elif self.orientation == 'clockwise':
            self.tangent_to_normal_rotation_matrix = np.array([[0,-1],[1,0]])
    
    #@tf.contrib.eager.defun #wrapper function for the RHS function to permit parallelization
    def RHS_evaluate(self, x):
        return tf.map_fn(self.RHS_function, x, dtype = self.dtype)
            
    def get_tangent_vectors(self,t_values, return_unit_vectors = False): #get tangent vectors to the boundary at the specified pts
        tmp = np.array([self.dxdt(t_values), self.dydt(t_values)])
        if return_unit_vectors:
            return np.divide(tmp, np.linalg.norm(tmp, axis = 0))
        else:
            return tmp
        
    def get_normal_vectors(self,t_values, return_unit_vectors = False): #get normal vectors to the boundary at the specified pts
        return np.einsum('ij,jk->ik', self.tangent_to_normal_rotation_matrix, self.get_tangent_vectors(t_values, return_unit_vectors = return_unit_vectors))    
    
    @staticmethod
    def get_arc_length(coordinates): #approximates arc length along the boundary for a curve fit
        return np.cumsum(np.insert(np.sqrt(np.sum(np.square(np.subtract(np.array(coordinates[1:]),np.array(coordinates[:-1]))), axis = 1)),0,0.0))
    
    def LHS_evaluate(self, model, t_values): #evaluates the LHS at the specified pts., LHS has to be a keras model or otherwise return a mapping (x,y)->u from a method named 'predict_on_batch' when given the coordinates desired as a 2d tensorflow tensor
         x = self.x_interpolant(t_values)
         y = self.y_interpolant(t_values)
         
         self.last_used_pts = tf.constant(np.array([x,y]).T, dtype = self.dtype)
         
         if self.boundary_type == 'Dirichlet':
             if self.derivative_regularizer_coefficient == 0.0:
                 return model.predict_on_batch(self.last_used_pts), None
             else: #optionally calculates du/dt if needed
                 with tf.GradientTape() as tape:
                     tape.watch(self.last_used_pts)
                     y = model.predict_on_batch(self.last_used_pts)
                 grad = tape.gradient(y, self.last_used_pts)
                 dydt = tf.einsum('ij,ij->i',grad,tf.constant(self.get_tangent_vectors(t_values, return_unit_vectors = True).T,dtype = self.dtype))
                 return y, dydt
                 
         elif self.boundary_type == 'von Neumann' or self.boundary_type == 'Robin':
             
             with tf.GradientTape(persistent=True) as tape:
                 tape.watch(self.last_used_pts)
                 y = model.predict_on_batch(self.last_used_pts)
             grad = tape.gradient(y, self.last_used_pts)
             normal_gradients = tf.einsum('ij,ij->i',grad,tf.constant(self.get_normal_vectors(t_values, return_unit_vectors = True).T,dtype = self.dtype))
             if self.derivative_regularizer_coefficient == 0.0:
                 dydt = None
             else:
                 dydt = tf.einsum('ij,ij->i',grad,tf.constant(self.get_tangent_vectors(t_values, return_unit_vectors = True).T,dtype = self.dtype))
             
             if self.boundary_type == 'von Neumann':
                 return normal_gradients, dydt
             elif self.boundary_type == 'Robin':
                 return normal_gradients + self.robin_bc_alpha * y, dydt
    
    def evaluate_error(self, model, n_quadpts = 5): #evaluates error
        #Evaluates the error between the values provided by the model and the RHS using Gauss-Legendre quadrature
        quadrature_t, quadrature_w = tuple([np.polynomial.legendre.leggauss(n_quadpts)[i].astype(self.dtype) for i in range(2)]) #get quadrature pts
        c = np.array(0.5*(self.t[-1] - self.t[0]),dtype=self.dtype)#helper variables
        d = np.array(0.5*(self.t[-1] + self.t[0]),dtype=self.dtype)
        quadpts = tf.constant(c*quadrature_t+d, dtype = self.dtype) #quadrature points on boundary
        #pdb.set_trace()
        LHS, LHSderivatives = self.LHS_evaluate(model, quadpts) #LHS values
        
        #RHS values
        if self.boundary_rhs_is_parametric:
            if self.derivative_regularizer_coefficient == 0.0:
                RHS = self.RHS_evaluate(quadpts)
            else:
                with tf.GradientTape() as tape:
                    tape.watch(quadpts)
                    RHS = self.RHS_evaluate(quadpts)
                RHSderivatives = tape.gradient(RHS, quadpts)
        else:
            if self.derivative_regularizer_coefficient == 0.0:
                RHS = self.RHS_evaluate(quadpts)
            else:
                with tf.GradientTape() as tape:
                    tape.watch(self.last_used_pts)
                    RHS = self.RHS_evaluate(quadpts)
                RHSderivatives = tf.einsum('ij,ij->i', tape.gradient(RHS, self.last_used_pts), tf.constant(self.get_tangent_vectors(quadpts, return_unit_vectors = True).T,dtype = self.dtype))
        
        #Calculate Lp norm and return
        if len(self.error_type) == 2 and self.error_type[0] == 'L':
            Lp_norm_order = int(self.error_type[1])
            error_function = tf.pow(tf.reduce_sum(tf.multiply(c*quadrature_w,tf.pow(tf.reshape(LHS, [tf.reduce_prod(LHS.shape)])-RHS,Lp_norm_order))),1/Lp_norm_order)
            if LHSderivatives != None:
                error_derivatives = tf.pow(tf.reduce_sum(tf.multiply(c*quadrature_w,tf.pow(LHSderivatives-RHSderivatives,Lp_norm_order))),1/Lp_norm_order)
            else:
                error_derivatives = 0.0
            return error_function + self.derivative_regularizer_coefficient * error_derivatives
                
