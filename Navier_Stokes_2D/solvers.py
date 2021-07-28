# -*- coding: utf-8 -*-
"""
This file contains the iterative numerical solvers which uses Projection methods
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator
import scipy.sparse
import scipy.sparse.linalg as slg
from pyamg import smoothed_aggregation_solver
from matplotlib import cm
import time
import sys
import copy
import structure

import tensorflow as tf
import poisson_CNN
import itertools
#import pyamg
import json
import matplotlib as mpl
mpl.rcParams['figure.dpi']=300

#__all__ = ['LinearSystem_solver', 'Gauge_method', 'Alg1', 'Error']

cfg = poisson_CNN.convert_tf_object_names(json.load(open('/poisson_CNN/poisson_CNN/experiments/hpnn_neumann_piloss_smalldomain.json')))
model = poisson_CNN.models.Homogeneous_Poisson_NN_Legacy(**cfg['model'])
_ = model([tf.random.uniform((2,1,100,100)), tf.random.uniform((2,1))])
model.compile(loss='mse', optimizer = 'adam')
model.load_weights('/storage/training-results/hpnn_legacy_neumann_smallstencil_smalldomain/chkpt.checkpoint')

class LinearSystem_solver():
    '''this class contains the linear system solvers for both velocity and pressure
	it returns the linear system in Scipy sparse matrix form and linear operator form'''
    mod = model
    _timestep_counter = itertools.count(0)
    def __init__(self, Re, mesh, integration_method='Riemann'):
        
        self._images_folder = '/storage/Navier_Stokes_2D/plots_zero+bicgstab2/'
        #self.mod = mod
        self.mesh = mesh
        self.Re = Re
        self.integration_method = integration_method

    # Linear systemas for velocities (in the form of sparse matrices)
    # It can be used for both intermediate velocity fields (u*) and Gauge variables (m)
    # It returns both the sparse matrix system A and its linear operator 
    def Linsys_velocity_matrix(self, velocity):
        m = self.mesh.m
        n = self.mesh.n
        dt = self.mesh.dt
        dx = self.mesh.dx
        dy = self.mesh.dy
        Re = self.Re
        # for square domain only, lx = ly and dx = dy = dh
        dh = dx
        a = dt/(2*Re*dh**2)
        b = (Re*dh**2)/dt + 2

        # Dirichlet boundary condition is applied
        if velocity == "u":
            # construct matrix A: Au = rhs
            # A is symmetric and positive definite with dimension NxN
            N = m*(n-1)
            # block matrix
            maindiag = np.zeros(n-1)
            maindiag[:] = 2*b
            sidediag = np.zeros(n-2)
            sidediag[:] = -1
            B = scipy.sparse.diags([maindiag,sidediag,sidediag],[0,-1,1])
            A1 = scipy.sparse.kron(scipy.sparse.eye(m,m),B)
            md = np.zeros(N)
            md[0:n-1] = 3.0
            md[-(n-1):] = 3.0
            sdl = -np.ones(N-(n-1))
            sdl[-(n-1):] = -2.0
            sdu = sdl[::-1]
            sdll = np.zeros((n-2)*(n-1))
            sdll[-(n-1):] = 0.2
            sduu = sdll[::-1]
            A2 = scipy.sparse.diags([md,sdl,sdu,sdll,sduu],[0,-(n-1),n-1,-2*(n-1),2*(n-1)])
            A = scipy.sparse.csc_matrix((A1+A2)*a)
            #print np.linalg.cond(np.matrix(A.todense())), "condition number velocity"
            A_linop = scipy.sparse.linalg.aslinearoperator(A)
            return [A, A_linop]

        elif velocity == "v":
            # construct A: Av = rhs
            N = (m-1)*n
            # block matrix
            maindiag = np.zeros(n)
            maindiag[:] = 2*b
            maindiag[0] = 2*b+3
            maindiag[-1] = 2*b+3
            sidediagl = -np.ones(n-1)
            sidediagl[-1] = -2.0
            sidediagu = sidediagl[::-1]
            sdl = np.zeros(n-2)
            sdl[-1] = 0.2
            sdu = sdl[::-1]
            B = scipy.sparse.diags([maindiag,sidediagl,sidediagu,sdl,sdu],[0,-1,1,-2,2])
            A1 = scipy.sparse.kron(scipy.sparse.eye(m-1,m-1),B)
            sd = -np.ones(N-n)
            A2 = scipy.sparse.diags([sd,sd],[-n,n])
            A = scipy.sparse.csc_matrix((A1+A2)*a)
            #print np.linalg.cond(np.matrix(A.todense())), "condition number velocity"
            A_linop = scipy.sparse.linalg.aslinearoperator(A)

            return [A,A_linop]

    # the linear system solver for velocity fields (using Biconjugate gradient method)
    # returns VelocityField instances (only interior points are calculated)
    # ALuv = [A, A_linop]: contains the lineary system in the sparse matrix and linear operator form
    # rhsuv = [rhsu, rhsv]: right hand side of u and v velocities (they need to be boundary corrected)
    def Linsys_velocity_solver(self, ALuv, rhsuv, tol=1e-12):
        m = self.mesh.m
        n = self.mesh.n
        dx = self.mesh.dx
        dy = self.mesh.dy
        # for square domain only, lx = ly and dx = dy = dh
        dh = dx
        uvl = []
        # only solving the interior points, rhsuv needs to be boundary corrected
        ## solve for u and v sequentially
        for i in range(2):
            ## for u
            if i == 0:
                N = m*(n-1)
                row = m
                col = n-1
            ## for v
            else:
                 N = (m-1)*n
                 row = m-1
                 col = n
            ## convert rhs into vector (m*(n-1))
            rhs = rhsuv.get_uv()[i]
            rhs = rhs.reshape(N)
            AL = ALuv[i]                
            A = AL[0]
            A_linop = AL[1]
            u = scipy.sparse.linalg.bicg(A=A_linop, b=rhs, tol=tol)
            u = u[0].reshape(row, col)
            uvl.append(u)
            AL = []
            rhs = 0          
            row = 0
            col = 0
        # uvstar: u* the intermediate velocity field in the form of VelocityField object
        # note that this is the same as the Gauge variable (m) in the Gauge method
        uvstar = structure.VelocityField(uvl[0], uvl[1], self.mesh)
        return uvstar

    # the Pressure Poisson lineary system
    # returns thePoisson pressure matrix A, preconditioner and its linear operaters (if applicable)
    def Poisson_pressure_matrix(self, solve_method):
        m = self.mesh.m
        n = self.mesh.n
        dx = self.mesh.dx
        dy = self.mesh.dy
        # for square domain only, lx = ky and dx = dy = dh
        dh = dx
        # construct matrix A: Ap = rhs, p is pressure (with interior points)
        # Neumann boundary condition is applied
        # A is negative definite so use -A which is positive definite
        # block matrix                       
        maindiag = np.ones(n)
        maindiag[1:n-1] = (2*maindiag[1:n-1])
        sidediag = np.ones(n-1)
        B = scipy.sparse.diags([maindiag/(dh**2),-sidediag/(dh**2),-sidediag/(dh**2)],[0,-1,1])
        A1 = scipy.sparse.kron(scipy.sparse.eye(m,n),B)
        A2 = scipy.sparse.kron(B, scipy.sparse.eye(m,n))
        A = A1+A2
        A = scipy.sparse.csc_matrix(A)
        # add the zero integral constraint
        # integration matrix
        C = self.mesh.integrate(integration_method=self.integration_method)
        A = scipy.sparse.hstack([A,scipy.sparse.csc_matrix(np.matrix(C).T)])
        # add one zero column to make sure A is square
        C = np.append(C,0)
        A = scipy.sparse.vstack([A,scipy.sparse.csc_matrix(C)])
        A = scipy.sparse.csc_matrix(A)
        #print np.linalg.cond(A), 'condition number of the Poisson pressure linear system solver

        # Biconjugate gradient method
        if solve_method == "ILU":
            A_linop = scipy.sparse.linalg.aslinearoperator(A)
            # MMD_AT_PLUS_A, MMD_ATA, COLAMD defines different types of preconditioners
            # for more detail, see Scipy.sparse.linalg.spilu documentations
            A_ILU = slg.spilu(A,permc_spec='MMD_AT_PLUS_A')
            #A_ILU = slg.spilu(A,permc_spec='MMD_ATA')
            #A_ILU = slg.spilu(A,permc_spec='COLAMD')
            M = slg.LinearOperator(shape=(m*n+1,m*n+1),matvec=A_ILU.solve)
            return [A_linop, M, A]

        # direct solve
        elif solve_method == "DIR":
            return A

    # Solves the Pressure Poisson problem using either Biconjugate gradient method (with ILU factorisation preconditioner) or direct solve
    def Poisson_pressure_solver(self, rhs, solve_method, precd_AL, tol=1e-12):
        m = self.mesh.m
        n = self.mesh.n
        dt = self.mesh.dt
        dx = self.mesh.dx
        dy = self.mesh.dy
        extent = (0.0, (m - 1) * dx, 0.0, (n-1) * dy)
        ts = next(self._timestep_counter)
        # for square domain only, lx = ky and dx = dy = dh
        dh = dx
        N = m*n
        
        # compute and plot ground truth
        if solve_method == "ILU":
            _rhs = (-rhs).get_value().reshape(m*n)
            _rhs = np.hstack([_rhs, np.zeros(1)])
            # use Incomplete LU to find a preconditioner
            A_linop = precd_AL[0]
            M = precd_AL[1]
            A = precd_AL[2]
            _p = scipy.sparse.linalg.bicgstab(A=A_linop, b=_rhs, tol=tol, maxiter=N, M=M)[0]
            Ap = A*np.matrix(np.ravel(_p)).T
            r = _rhs - np.array(Ap.T)
            #print(np.max(np.abs(r)), "residual")
            #print(_p[-1], 'lambda constant')
            _p = _p[:-1]
            _p = _p.reshape(m,n)
            maxmag = np.max(np.abs(_p))
            vmax, vmin = maxmag, -maxmag
            plt.imshow(_p, extent = extent, vmax = vmax, vmin = vmin, cmap = 'RdBu', origin = 'lower')
            plt.colorbar()
            plt.savefig(self._images_folder + 'gt_' + str(ts) +'.png', bbox_inches = 'tight')
            plt.close()
            # p = structure.CentredPotential(p, self.mesh)
            # print(self.mesh.integrate(p, self.integration_method), 'integral of phi')
            # returns p (phi) variable in the form of CentredPotential object
            # return p

        # convert rhs into vector (m*n)
        rhs = rhs.get_value()
        # trhs, sf = poisson_CNN.utils.set_max_magnitude_in_batch_and_return_scaling_factors(rhs.reshape([1,1] + list(rhs.shape)).astype(np.float32), 1.0)
        
        # tdx = tf.cast(tf.constant([[dx]]), tf.float32)
        # trhs = tf.cast(trhs, tf.float32)
        # pred = self.mod([trhs, tdx])
        # pred = (((dx * (n-1))**2) / sf) * pred
        
        # plt.imshow(pred[0,0], extent = extent, vmax = vmax, vmin = vmin, cmap = 'RdBu', origin = 'lower')
        # plt.colorbar()
        # plt.savefig(self._images_folder + 'pred_' + str(ts) +'.png', bbox_inches = 'tight')
        # plt.close()
        
        rhs = (-rhs).reshape(m*n)
            
        # add the zero integration constraint to the right hand side
        rhs = np.hstack([rhs, np.zeros(1)])
        
        
        #x0 = np.hstack([pred.numpy().reshape(m*n), np.zeros(1)])
        x0 = np.zeros(m*n+1)
        A_linop = precd_AL[0]
        M = precd_AL[1]
        A = precd_AL[2]
        p = scipy.sparse.linalg.bicgstab(A=A_linop, b=rhs, x0 = x0, tol=tol, maxiter=2, M=M)[0]
        Ap = A*np.matrix(np.ravel(p)).T
        r = rhs - np.array(Ap.T)
        print(np.max(np.abs(r)), "residual")
        print(p[-1], 'lambda constant')
        p = p[:-1]
        p = p.reshape(m,n)
        
        plt.imshow(p, extent = extent, vmax = vmax, vmin = vmin, cmap = 'RdBu', origin = 'lower')
        plt.colorbar()
        plt.savefig(self._images_folder + 'pred+bicgstab1_' + str(ts) +'.png', bbox_inches = 'tight')
        plt.close()
        '''
        p = scipy.sparse.linalg.bicgstab(A=A_linop, b=rhs, tol=tol, maxiter=2, M=M)[0]
        Ap = A*np.matrix(np.ravel(p)).T
        r = rhs - np.array(Ap.T)
        print(np.max(np.abs(r)), "residual")
        print(p[-1], 'lambda constant')
        p = p[:-1]
        p = p.reshape(m,n)
        plt.imshow(p)
        plt.colorbar()
        plt.savefig('zeroinitial+bicgstab2.png')
        plt.close()
        

        # Biconjugate gradient method
        if solve_method == "ILU":
            # use Incomplete LU to find a preconditioner
            A_linop = precd_AL[0]
            M = precd_AL[1]
            A = precd_AL[2]
            p = scipy.sparse.linalg.bicgstab(A=A_linop, b=rhs, tol=tol, maxiter=N, M=M)[0]
            Ap = A*np.matrix(np.ravel(p)).T
            r = rhs - np.array(Ap.T)
            print(np.max(np.abs(r)), "residual")
            print(p[-1], 'lambda constant')
            p = p[:-1]
            p = p.reshape(m,n)
            # p = structure.CentredPotential(p, self.mesh)
            # print(self.mesh.integrate(p, self.integration_method), 'integral of phi')
            # returns p (phi) variable in the form of CentredPotential object
            # return p
        
        # direct solve
        elif solve_method == "DIR":
            A = precd_AL
            p = scipy.sparse.linalg.spsolve(A=A, b=rhs)
            Ap = A*np.matrix(np.ravel(p)).T
            r = rhs - np.array(Ap.T)
            print(np.max(np.abs(r)), "residual")
            print(p[-1], 'lambda constant')
            p = p[:-1]
            p = p.reshape(m,n)
            # print(np.sum(p), 'integral of phi')
            # p = structure.CentredPotential(p, self.mesh)

        plt.imshow(p)
        plt.colorbar()
        plt.savefig('gt.png')
        plt.close()
        import pdb
        pdb.set_trace()
        '''

        # returns p (phi) variable in the form of CentredPotential object
        p = structure.CentredPotential(p, self.mesh)
        return p

# below constructs the 4 different Projection method solvers (Gauge, Alg 1, Alg 2, Alg 3)
class Gauge_method():
    '''This class constructs the Gauge method solver'''

    def __init__(self, Re, mesh):
        self.Re = Re
        self.n = mesh.n
        self.m = mesh.m
        self.xu = mesh.xu
        self.yu = mesh.yu
        self.xv = mesh.xv
        self.yv = mesh.yv
        self.gds = mesh.gds
        self.sdomain = mesh.sdomain
        self.tdomain = mesh.tdomain
        self.Tn = mesh.Tn
        self.t0 = mesh.tdomain[0]
        self.dt = mesh.dt
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.mesh = mesh

    # initial set up
    def setup(self, InCond_uv_init, Boundary_uv_type, solve_method='ILU', integration_method='Riemann'):
        ## InCond_uv: specifies the velocity initial condition 
        linsys_solver = LinearSystem_solver(self.Re, self.mesh, integration_method)
        phi_mat = linsys_solver.Poisson_pressure_matrix(solve_method)
        m1_mat = linsys_solver.Linsys_velocity_matrix("u")
        m2_mat = linsys_solver.Linsys_velocity_matrix("v")

        InCond_uvcmp = structure.VelocityComplete(self.mesh, InCond_uv_init, 0).complete(Boundary_uv_type)
        uv_cmp = copy.copy(InCond_uvcmp)        
        mn_cmp = copy.copy(uv_cmp)
        initial_setup_parameters = [phi_mat, m1_mat, m2_mat, InCond_uvcmp, uv_cmp, mn_cmp, integration_method, solve_method]
        return initial_setup_parameters

    def iterative_solver(self, Boundary_uv_type, Tn, initial_setup_parameters):
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt
        Re = self.Re
        phi_mat = initial_setup_parameters[0]
        m1_mat = initial_setup_parameters[1]
        m2_mat = initial_setup_parameters[2]
        # uvold_cmp: u and v velocity fields at time n-1
        # cmp: in the completed format (interior + boundary + ghost nodes)
        uvold_cmp = initial_setup_parameters[3]
        # uv_cmp: u and v at time n
        uv_cmp = initial_setup_parameters[4]
        # Gauge variable at time n (in the completed format)
        mn_cmp = initial_setup_parameters[5]
        integration_method = initial_setup_parameters[6]
        solve_method = initial_setup_parameters[7] 
        # int: interior points only
        mn_int = structure.VelocityField(mn_cmp.get_int_uv()[0], mn_cmp.get_int_uv()[1], self.mesh)
        # phiold: phi variable at time n-1
        phiold = np.zeros((m,n))
        phiold_cmp = structure.CentredPotential(phiold, self.mesh).complete()
        # phin_cmp: phi variable at time n
        phin_cmp = np.copy(phiold_cmp)

        print(Tn, "number of iterations")
        # main iterative solver
        test_problem_name = Boundary_uv_type
        for t in range(Tn):
            forcing_term = structure.Forcing_term(self.mesh, test_problem_name, t+0.5).select_forcing_term()
            convc_uv = uv_cmp.non_linear_convection()
            preconvc_uv = uvold_cmp.non_linear_convection()
            diff_mn = mn_cmp.diffusion()

            if Boundary_uv_type == 'periodic_forcing_1':
                # Stokes problem
                rhs_mstar = mn_int + dt*((1.0/(2*Re))*diff_mn + forcing_term)  
            elif Boundary_uv_type == 'periodic_forcing_2':
                # Stokes problem
                rhs_mstar = mn_int + dt*((1.0/(2*Re))*diff_mn + forcing_term)	
            else:
                # full Navier Stokes problem
                rhs_mstar = mn_int + dt*(-1.5*convc_uv + 0.5*preconvc_uv + (1.0/(2*Re))*diff_mn + forcing_term) 

            # calculate the approximation to phi at time n+1
            gradphiuv = self.gradphi_app(phiold_cmp, phin_cmp)
            # boundary correction step
            rhs_mstarcd = self.correct_boundary(rhs_mstar, t+1, Boundary_uv_type, gradphiuv)
            # solving for the Gauge variable m
            Linsys_solve = LinearSystem_solver(Re, self.mesh)
            mstar = Linsys_solve.Linsys_velocity_solver([m1_mat,m2_mat],  rhs_mstarcd)
            mstarcmp1, uvbnd_value = structure.VelocityComplete(self.mesh, [mstar.get_uv()[0],  mstar.get_uv()[1]], t+1).complete(Boundary_uv_type, return_bnd=True)
            div_mstar = mstarcmp1.divergence()
            # solving for the phi variable
            phi = Linsys_solve.Poisson_pressure_solver(div_mstar, solve_method, phi_mat)
            print(solve_method)
            if t == 0:
                #div_mn = np.zeros((m,n))
                div_mn = div_mstar
            else:
                div_mn = mn_cmp.divergence()

            phiacd = phi - phin_cmp[1:m+1,1:n+1]
            # pressure correction step
            p = phiacd/dt - 1.0/(2*Re)*(div_mstar+div_mn)
            print(self.mesh.integrate(p, integration_method), 'integral of p')
            gradp = p.gradient()
            phiold_cmp = np.copy(phin_cmp)
            phin_cmp = np.copy(phi.complete())
            # velocity update step
            gradphi = phi.gradient()
            uvn_int = mstar - gradphi
            uvold_cmp = copy.copy(uv_cmp)
            uv_cmp = structure.VelocityComplete(self.mesh, [uvn_int.get_uv()[0],  uvn_int.get_uv()[1]], t+1).complete(Boundary_uv_type)
            # complete mstar
            mn_cmp = self.complete_mstar(mstar, uvbnd_value, phin_cmp)
            mn_int = structure.VelocityField(mn_cmp.get_int_uv()[0], mn_cmp.get_int_uv()[1], self.mesh)            
            print("iteration "+str(t))
        return uv_cmp, p, gradp

    ## this function calculates graident of phi at time n+1
    # using second order approximation to gradient of phi^(n+1). Used in correcting m*
    # phi^{n+1} appro 2*phi^n - phi^{n-1}
    def gradphi_app(self, phiold_cmp, phin_cmp):
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt

        phiapp_cmp = 2*phin_cmp - phiold_cmp
        gradphiu = (phiapp_cmp[:,1:n+2] - phiapp_cmp[:,0:n+1])/dx
        gradphiv = (phiapp_cmp[1:m+2,:] - phiapp_cmp[0:m+1,:])/dy
        # obtain gradphiu North and South boundary by cubic interpolation
        gradphiuN = 5.0/16*(gradphiu[0,:] +3*gradphiu[1,:] - gradphiu[2,:]+0.2*gradphiu[3,:])
        gradphiuS = 5.0/16*(gradphiu[-1,:] +3*gradphiu[-2,:] - gradphiu[-3,:]+0.2*gradphiu[-4,:])
        gradphiu[0,:] = gradphiuN
        gradphiu[-1,:] = gradphiuS

        # obtain gradphiv West and East boundary by cubic interpolation
        gradphivW = 5.0/16*(gradphiv[:,0] +3*gradphiv[:,1] - gradphiv[:,2]+0.2*gradphiv[:,3])
        gradphivE = 5.0/16*(gradphiv[:,-1] +3*gradphiv[:,-2] - gradphiv[:,-3]+0.2*gradphiv[:,-4])
        gradphiv[:,0] = gradphivW
        gradphiv[:,-1] = gradphivE
        return [gradphiu, gradphiv]

    # boundary correction used in solving for Gauge variable
    def correct_boundary(self, rhs_mstar, t, Boundary_type, gradphiuv):
        # rhsuv is a VelocityField object with dimension interior u and v [(m*(n-1), (m-1)*n)]
        n = self.n
        m = self.m
        Re = self.Re
        dx = self.dx
        dy = self.dy
        dt = self.dt

        lam = dt/(2.0*Re)
        VC = structure.VelocityComplete(self.mesh, [rhs_mstar.get_uv()[0], rhs_mstar.get_uv()[1]], t)
        gradphiu = gradphiuv[0]
        gradphiv = gradphiuv[1]

        if Boundary_type == "driven_cavity":
            uN = VC.bnd_driven_cavity('u')['N']
            uS = VC.bnd_driven_cavity('u')['S']
            uW = VC.bnd_driven_cavity('u')['W']
            uE = VC.bnd_driven_cavity('u')['E']

            vN = VC.bnd_driven_cavity('v')['N']
            vS = VC.bnd_driven_cavity('v')['S']
            vW = VC.bnd_driven_cavity('v')['W']
            vE = VC.bnd_driven_cavity('v')['E']

        elif Boundary_type == "Taylor":
            uN = VC.bnd_Taylor('u')['N'][1:n]
            uS = VC.bnd_Taylor('u')['S'][1:n]
            uW = VC.bnd_Taylor('u')['W']
            uE = VC.bnd_Taylor('u')['E']

            vN = VC.bnd_Taylor('v')['N']
            vS = VC.bnd_Taylor('v')['S']
            vW = VC.bnd_Taylor('v')['W'][1:m]
            vE = VC.bnd_Taylor('v')['E'][1:m]
        elif Boundary_type == "periodic_forcing_1":
            uN = VC.bnd_forcing_1('u')['N'][1:n]
            uS = VC.bnd_forcing_1('u')['S'][1:n]
            uW = VC.bnd_forcing_1('u')['W']
            uE = VC.bnd_forcing_1('u')['E']

            vN = VC.bnd_forcing_1('v')['N']
            vS = VC.bnd_forcing_1('v')['S']
            vW = VC.bnd_forcing_1('v')['W'][1:m]
            vE = VC.bnd_forcing_1('v')['E'][1:m]

        elif Boundary_type == "periodic_forcing_2":
            uN = VC.bnd_forcing_2('u')['N'][1:n]
            uS = VC.bnd_forcing_2('u')['S'][1:n]
            uW = VC.bnd_forcing_2('u')['W']
            uE = VC.bnd_forcing_2('u')['E']

            vN = VC.bnd_forcing_2('v')['N']
            vS = VC.bnd_forcing_2('v')['S']
            vW = VC.bnd_forcing_2('v')['W'][1:m]
            vE = VC.bnd_forcing_2('v')['E'][1:m]

        gradphiuW = gradphiu[1:m+1,0]
        gradphiuE = gradphiu[1:m+1,-1]
        gradphiuN = gradphiu[0,1:n]
        gradphiuS = gradphiu[-1,1:n]

        # North and South boundary
        uNbc = uN + gradphiuN
        uSbc = uS + gradphiuS

        resu1 = np.zeros((m,n-1))
        resu2 = np.zeros((m,n-1))
        resu1[0,:] = (16.0/5)*(uNbc)*(lam/(dy**2))
        resu1[-1,:] = (16.0/5)*(uSbc)*(lam/(dy**2))            

        # West and East boundary
        uWbc = uW
        uEbc = uE
        resu2[:,0] = (uWbc)*(lam/(dx**2))
        resu2[:,-1] = (uEbc)*(lam/(dx**2))
        resu = resu1+resu2

        resv1 = np.zeros((m-1,n))
        resv2 = np.zeros((m-1,n))

        gradphivN = gradphiv[0,1:n+1]
        gradphivS = gradphiv[-1,1:n+1]
        gradphivW = gradphiv[1:m,0]
        gradphivE = gradphiv[1:m,-1]

        # North and South boundary
        vNbc = vN
        vSbc = vS
        resv2[0,:] = vNbc*(lam/(dy**2))
        resv2[-1,:] = vSbc*(lam/(dy**2))

        # West and East boundary
        vWbc = vW + gradphivW
        vEbc = vE + gradphivE
        resv1[:,0] = (16.0/5)*vWbc*(lam/(dx**2))
        resv1[:,-1] = (16.0/5)*vEbc*(lam/(dx**2))

        resv = resv1+resv2
        rhs_mstarcd = rhs_mstar + [resu, resv]

        return rhs_mstarcd

    # completing the Gauge variable at time n+1 
    def complete_mstar(self, mstar_int, uvbnd_value, phiacd_cmp):
        # complete m* using phi^(n+1)
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt    
        uN, uS, uW, uE = uvbnd_value[0]
        vN, vS, vW, vE = uvbnd_value[1]

        m1star_cmp = np.zeros((m+2,n+1))
        m2star_cmp = np.zeros((m+1,n+2))
        m1star_cmp[1:m+1,1:n] = mstar_int.get_uv()[0]
        m2star_cmp[1:m,1:n+1] = mstar_int.get_uv()[1]        
        m1star_cmp[1:m+1,0] = uW
        m1star_cmp[1:m+1,-1] = uE
        m2star_cmp[0,1:n+1] = vN
        m2star_cmp[-1,1:n+1] = vS        

        gdphi_cmpu = (phiacd_cmp[:,1:n+2] - phiacd_cmp[:,0:n+1])/dx
        gdphi_cmpuN = 5.0/16*(gdphi_cmpu[0,:] +3*gdphi_cmpu[1,:] - gdphi_cmpu[2,:]+0.2*gdphi_cmpu[3,:])
        gdphi_cmpuS = 5.0/16*(gdphi_cmpu[-1,:] +3*gdphi_cmpu[-2,:] - gdphi_cmpu[-3,:]+0.2*gdphi_cmpu[-4,:])

        # use phi^{n+1} just computed
        m1starN = uN + gdphi_cmpuN
        m1starS = uS + gdphi_cmpuS

        m1star_cmp[0,:] = (16.0/5)*m1starN - 3*m1star_cmp[1,:] + m1star_cmp[2,:] - 0.2*m1star_cmp[3,:]
        m1star_cmp[-1,:] = (16.0/5)*m1starS - 3*m1star_cmp[-2,:] + m1star_cmp[-3,:] - 0.2*m1star_cmp[-4,:]

        gdphi_cmpv = (phiacd_cmp[1:m+2,:] - phiacd_cmp[0:m+1,:])/dy
        gdphi_cmpvW = 5.0/16*(gdphi_cmpv[:,0] +3*gdphi_cmpv[:,1] - gdphi_cmpv[:,2]+0.2*gdphi_cmpv[:,3])
        gdphi_cmpvE = 5.0/16*(gdphi_cmpv[:,-1] +3*gdphi_cmpv[:,-2] - gdphi_cmpv[:,-3]+0.2*gdphi_cmpv[:,-4])
        m2starW = vW + gdphi_cmpvW
        m2starE = vE + gdphi_cmpvE
        m2star_cmp[:,0] = (16.0/5)*m2starW - 3*m2star_cmp[:,1] + m2star_cmp[:,2] - 0.2*m2star_cmp[:,3]
        m2star_cmp[:,-1] = (16.0/5)*m2starE - 3*m2star_cmp[:,-2] + m2star_cmp[:,-3] - 0.2*m2star_cmp[:,-4]

        return structure.VelocityField(m1star_cmp, m2star_cmp, self.mesh)
        
class Alg1_method():
    '''This class constructs the Alg 1 method solver
       Note that this solver is inherently first order accurate in time for the pressure variable because its pressure update formula limits the accuracy'''

    def __init__(self, Re, mesh):
        self.Re = Re
        self.n = mesh.n
        self.m = mesh.m
        self.xu = mesh.xu
        self.yu = mesh.yu
        self.xv = mesh.xv
        self.yv = mesh.yv
        self.gds = mesh.gds
        self.sdomain = mesh.sdomain
        self.tdomain = mesh.tdomain
        self.Tn = mesh.Tn
        self.t0 = mesh.tdomain[0]
        self.dt = mesh.dt
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.mesh = mesh
    
    # initial set up
    def setup(self, InCond, Boundary_uv_type, solve_method='ILU', integration_method='Riemann'):
        ## InCond_uv: specifies the velocity initial condition 
        linsys_solver = LinearSystem_solver(self.Re, self.mesh, integration_method)
        phi_mat = linsys_solver.Poisson_pressure_matrix(solve_method)
        u_mat = linsys_solver.Linsys_velocity_matrix("u")
        v_mat = linsys_solver.Linsys_velocity_matrix("v")

        InCond_uvcmp = structure.VelocityComplete(self.mesh, InCond[0], 0).complete(Boundary_uv_type)
        uvn_cmp = copy.copy(InCond_uvcmp)
        InCond_p = structure.CentredPotential(InCond[1], self.mesh)
        initial_setup_parameters = [phi_mat, u_mat, v_mat, InCond_uvcmp, uvn_cmp, InCond_p, integration_method, solve_method]
        return initial_setup_parameters

    def iterative_solver(self, Boundary_uv_type, Tn, initial_setup_parameters):
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt
        Re = self.Re
        phi_mat = initial_setup_parameters[0]
        u_mat = initial_setup_parameters[1]
        v_mat = initial_setup_parameters[2]
        # uvold_cmp: u and v velocity fields at time n-1
        # cmp: in the completed format (interior + boundary + ghost nodes)
        uvold_cmp = initial_setup_parameters[3]
        # uvn_cmp: u and v at time n
        uvn_cmp = initial_setup_parameters[4]
        pold = initial_setup_parameters[5]
        integration_method = initial_setup_parameters[6]
        solve_method = initial_setup_parameters[7]
        pn = copy.copy(pold)

        print(Tn, "number of iterations")
        # main iterative solver
        test_problem_name = Boundary_uv_type
        for t in range(Tn):
            forcing_term = structure.Forcing_term(self.mesh,test_problem_name,t+0.5).select_forcing_term()
            convc_uv = uvn_cmp.non_linear_convection()
            preconvc_uv = uvold_cmp.non_linear_convection()
            diff_uvn = uvn_cmp.diffusion()
            gradp_uvn = pn.gradient()
            uvn_int = structure.VelocityField(uvn_cmp.get_int_uv()[0], uvn_cmp.get_int_uv()[1], self.mesh)
            if Boundary_uv_type == 'periodic_forcing_1':
                # Stokes problem
                rhs_uvstar = uvn_int + dt*(- gradp_uvn + (1.0/(2*Re))*diff_uvn + forcing_term)
            elif Boundary_uv_type == 'periodic_forcing_2':
                # Stokes problem
                rhs_uvstar = uvn_int + dt*(- gradp_uvn + (1.0/(2*Re))*diff_uvn + forcing_term)	
            else:
                # full Navier Stokes problem
                rhs_uvstar = uvn_int + dt*(-1.5*convc_uv + 0.5*preconvc_uv - gradp_uvn + (1.0/(2*Re))*diff_uvn + forcing_term) 

            # boundary correction step
            rhs_uvstarcd = self.correct_boundary(rhs_uvstar, t+1, Boundary_uv_type)

            # solving for the intermediate velocity variable uv* 
            Linsys_solve = LinearSystem_solver(Re, self.mesh)
            uvstar = Linsys_solve.Linsys_velocity_solver([u_mat,v_mat],  rhs_uvstarcd)
            uvstarcmp, uvbnd_value = structure.VelocityComplete(self.mesh, [uvstar.get_uv()[0],  uvstar.get_uv()[1]], t+1).complete(Boundary_uv_type, return_bnd=True)
            div_uvstar = uvstarcmp.divergence()

            # solving for the phi variable
            phi = Linsys_solve.Poisson_pressure_solver(div_uvstar/dt, solve_method, phi_mat)
            # pressure correction step
            # note this formula makes the perssure variable first order accurate in time
            p = pn + phi 
            print(self.mesh.integrate(p, integration_method), 'integral of p')
            gradp = p.gradient()
            pold = copy.copy(pn)
            pn = copy.copy(p)

            # velocity update step
            gradphi = phi.gradient()
            uvn_int = uvstar - dt*gradphi
            uvold_cmp = copy.copy(uvn_cmp)
            uvn_cmp = structure.VelocityComplete(self.mesh, [uvn_int.get_uv()[0],  uvn_int.get_uv()[1]], t+1).complete(Boundary_uv_type)
            print("iteration "+str(t))
        return uvn_cmp, p, gradp

    # boundary correction 
    def correct_boundary(self, rhs_uvstar, t, Boundary_type):
        # rhsuv is a VelocityField object with dimension interior u and v [(m*(n-1), (m-1)*n)]
        n = self.n
        m = self.m
        Re = self.Re
        dx = self.dx
        dy = self.dy
        dt = self.dt

        lam = dt/(2.0*Re)
        VC = structure.VelocityComplete(self.mesh, [rhs_uvstar.get_uv()[0], rhs_uvstar.get_uv()[1]], t)

        if Boundary_type == "driven_cavity":
            uN = VC.bnd_driven_cavity('u')['N']
            uS = VC.bnd_driven_cavity('u')['S']
            uW = VC.bnd_driven_cavity('u')['W']
            uE = VC.bnd_driven_cavity('u')['E']

            vN = VC.bnd_driven_cavity('v')['N']
            vS = VC.bnd_driven_cavity('v')['S']
            vW = VC.bnd_driven_cavity('v')['W']
            vE = VC.bnd_driven_cavity('v')['E']

        elif Boundary_type == "Taylor":
            uN = VC.bnd_Taylor('u')['N'][1:n]
            uS = VC.bnd_Taylor('u')['S'][1:n]
            uW = VC.bnd_Taylor('u')['W']
            uE = VC.bnd_Taylor('u')['E']

            vN = VC.bnd_Taylor('v')['N']
            vS = VC.bnd_Taylor('v')['S']
            vW = VC.bnd_Taylor('v')['W'][1:m]
            vE = VC.bnd_Taylor('v')['E'][1:m]
        elif Boundary_type == "periodic_forcing_1":
            uN = VC.bnd_forcing_1('u')['N'][1:n]
            uS = VC.bnd_forcing_1('u')['S'][1:n]
            uW = VC.bnd_forcing_1('u')['W']
            uE = VC.bnd_forcing_1('u')['E']

            vN = VC.bnd_forcing_1('v')['N']
            vS = VC.bnd_forcing_1('v')['S']
            vW = VC.bnd_forcing_1('v')['W'][1:m]
            vE = VC.bnd_forcing_1('v')['E'][1:m]

        elif Boundary_type == "periodic_forcing_2":
            uN = VC.bnd_forcing_2('u')['N'][1:n]
            uS = VC.bnd_forcing_2('u')['S'][1:n]
            uW = VC.bnd_forcing_2('u')['W']
            uE = VC.bnd_forcing_2('u')['E']

            vN = VC.bnd_forcing_2('v')['N']
            vS = VC.bnd_forcing_2('v')['S']
            vW = VC.bnd_forcing_2('v')['W'][1:m]
            vE = VC.bnd_forcing_2('v')['E'][1:m]

        # North and South boundary
        resu1 = np.zeros((m,n-1))
        resu2 = np.zeros((m,n-1))
        resu1[0,:] = (16.0/5)*uN*(lam/(dy**2))
        resu1[-1,:] = (16.0/5)*uS*(lam/(dy**2))            

        # West and East boundary
        resu2[:,0] = uW*(lam/(dx**2))
        resu2[:,-1] = uE*(lam/(dx**2))
        resu = resu1+resu2

        resv1 = np.zeros((m-1,n))
        resv2 = np.zeros((m-1,n))

        # North and South boundary
        resv2[0,:] = vN*(lam/(dy**2))
        resv2[-1,:] = vS*(lam/(dy**2))

        # West and East boundary
        resv1[:,0] = (16.0/5)*vW*(lam/(dx**2))
        resv1[:,-1] = (16.0/5)*vE*(lam/(dx**2))

        resv = resv1+resv2
        rhs_uvstarcd = rhs_uvstar + [resu, resv]

        return rhs_uvstarcd

class Alg2_method():
    '''This class constructs the Alg 2 method solver'''

    def __init__(self, Re, mesh):
        self.Re = Re
        self.n = mesh.n
        self.m = mesh.m
        self.xu = mesh.xu
        self.yu = mesh.yu
        self.xv = mesh.xv
        self.yv = mesh.yv
        self.gds = mesh.gds
        self.sdomain = mesh.sdomain
        self.tdomain = mesh.tdomain
        self.Tn = mesh.Tn
        self.t0 = mesh.tdomain[0]
        self.dt = mesh.dt
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.mesh = mesh
    
    # initial set up
    def setup(self, InCond, Boundary_uv_type, solve_method='ILU', integration_method='Riemann'):
        ## InCond_uv: specifies the velocity initial condition 
        linsys_solver = LinearSystem_solver(self.Re, self.mesh, integration_method)
        phi_mat = linsys_solver.Poisson_pressure_matrix(solve_method)
        u_mat = linsys_solver.Linsys_velocity_matrix("u")
        v_mat = linsys_solver.Linsys_velocity_matrix("v")

        InCond_uvcmp = structure.VelocityComplete(self.mesh, InCond[0], 0).complete(Boundary_uv_type)
        uvn_cmp = copy.copy(InCond_uvcmp)
        InCond_p = structure.CentredPotential(InCond[1], self.mesh)
        initial_setup_parameters = [phi_mat, u_mat, v_mat, InCond_uvcmp, uvn_cmp, InCond_p, integration_method, solve_method]
        return initial_setup_parameters

    def iterative_solver(self, Boundary_uv_type, Tn, initial_setup_parameters):
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt
        Re = self.Re
        phi_mat = initial_setup_parameters[0]
        u_mat = initial_setup_parameters[1]
        v_mat = initial_setup_parameters[2]
        # uvold_cmp: u and v velocity fields at time n-1
        # cmp: in the completed format (interior + boundary + ghost nodes)
        uvold_cmp = initial_setup_parameters[3]
        # uvn_cmp: u and v at time n
        uvn_cmp = initial_setup_parameters[4]
        pold = initial_setup_parameters[5]
        integration_method = initial_setup_parameters[6]
        solve_method = initial_setup_parameters[7]
        pn = copy.copy(pold)

        print(Tn, "number of iterations")
        # main iterative solver
        test_problem_name = Boundary_uv_type
        for t in range(Tn):
            forcing_term = structure.Forcing_term(self.mesh,test_problem_name,t+0.5).select_forcing_term()
            convc_uv = uvn_cmp.non_linear_convection()
            preconvc_uv = uvold_cmp.non_linear_convection()
            diff_uvn = uvn_cmp.diffusion()
            gradp_uvn = pn.gradient()
            uvn_int = structure.VelocityField(uvn_cmp.get_int_uv()[0], uvn_cmp.get_int_uv()[1], self.mesh)
            if Boundary_uv_type == 'periodic_forcing_1':
                # Stokes problem
                rhs_uvstar = uvn_int + dt*(- gradp_uvn + (1.0/(2*Re))*diff_uvn + forcing_term)  
            elif Boundary_uv_type == 'periodic_forcing_2':
                # Stokes problem
                rhs_uvstar = uvn_int + dt*(- gradp_uvn + (1.0/(2*Re))*diff_uvn + forcing_term)  
            else:
                # full Navier Stokes problem
                rhs_uvstar = uvn_int + dt*(-1.5*convc_uv + 0.5*preconvc_uv - gradp_uvn + (1.0/(2*Re))*diff_uvn + forcing_term) 

            # boundary correction step
            rhs_uvstarcd = self.correct_boundary(rhs_uvstar, t+1, Boundary_uv_type)
            # solving for the intermediate velocity variable uv* 
            Linsys_solve = LinearSystem_solver(Re, self.mesh)
            uvstar = Linsys_solve.Linsys_velocity_solver([u_mat,v_mat],  rhs_uvstarcd)
            uvstarcmp, uvbnd_value = structure.VelocityComplete(self.mesh, [uvstar.get_uv()[0],  uvstar.get_uv()[1]], t+1).complete(Boundary_uv_type, return_bnd=True)
            div_uvstar = uvstarcmp.divergence()

            # solving for the phi variable
            phi = Linsys_solve.Poisson_pressure_solver(div_uvstar/dt, solve_method, phi_mat)
            # pressure correction step
            p = pn + phi - div_uvstar/(2*Re)
            print(self.mesh.integrate(p, integration_method), 'integral of p')
            gradp = p.gradient()
            pold = copy.copy(pn)
            pn = copy.copy(p)
            # velocity update stemp
            gradphi = phi.gradient()
            uvn_int = uvstar - dt*gradphi
            uvold_cmp = copy.copy(uvn_cmp)
            uvn_cmp = structure.VelocityComplete(self.mesh, [uvn_int.get_uv()[0],  uvn_int.get_uv()[1]], t+1).complete(Boundary_uv_type)
            print("iteration "+str(t))
        return uvn_cmp, p, gradp

    # boundary correction 
    def correct_boundary(self, rhs_uvstar, t, Boundary_type):
        # rhsuv is a VelocityField object with dimension interior u and v [(m*(n-1), (m-1)*n)]
        n = self.n
        m = self.m
        Re = self.Re
        dx = self.dx
        dy = self.dy
        dt = self.dt

        lam = dt/(2.0*Re)
        VC = structure.VelocityComplete(self.mesh, [rhs_uvstar.get_uv()[0], rhs_uvstar.get_uv()[1]], t)

        if Boundary_type == "driven_cavity":
            uN = VC.bnd_driven_cavity('u')['N']
            uS = VC.bnd_driven_cavity('u')['S']
            uW = VC.bnd_driven_cavity('u')['W']
            uE = VC.bnd_driven_cavity('u')['E']

            vN = VC.bnd_driven_cavity('v')['N']
            vS = VC.bnd_driven_cavity('v')['S']
            vW = VC.bnd_driven_cavity('v')['W']
            vE = VC.bnd_driven_cavity('v')['E']

        elif Boundary_type == "Taylor":
            uN = VC.bnd_Taylor('u')['N'][1:n]
            uS = VC.bnd_Taylor('u')['S'][1:n]
            uW = VC.bnd_Taylor('u')['W']
            uE = VC.bnd_Taylor('u')['E']

            vN = VC.bnd_Taylor('v')['N']
            vS = VC.bnd_Taylor('v')['S']
            vW = VC.bnd_Taylor('v')['W'][1:m]
            vE = VC.bnd_Taylor('v')['E'][1:m]
        elif Boundary_type == "periodic_forcing_1":
            uN = VC.bnd_forcing_1('u')['N'][1:n]
            uS = VC.bnd_forcing_1('u')['S'][1:n]
            uW = VC.bnd_forcing_1('u')['W']
            uE = VC.bnd_forcing_1('u')['E']

            vN = VC.bnd_forcing_1('v')['N']
            vS = VC.bnd_forcing_1('v')['S']
            vW = VC.bnd_forcing_1('v')['W'][1:m]
            vE = VC.bnd_forcing_1('v')['E'][1:m]

        elif Boundary_type == "periodic_forcing_2":
            uN = VC.bnd_forcing_2('u')['N'][1:n]
            uS = VC.bnd_forcing_2('u')['S'][1:n]
            uW = VC.bnd_forcing_2('u')['W']
            uE = VC.bnd_forcing_2('u')['E']

            vN = VC.bnd_forcing_2('v')['N']
            vS = VC.bnd_forcing_2('v')['S']
            vW = VC.bnd_forcing_2('v')['W'][1:m]
            vE = VC.bnd_forcing_2('v')['E'][1:m]

        # North and South boundary
        resu1 = np.zeros((m,n-1))
        resu2 = np.zeros((m,n-1))
        resu1[0,:] = (16.0/5)*uN*(lam/(dy**2))
        resu1[-1,:] = (16.0/5)*uS*(lam/(dy**2))            

        # West and East boundary
        resu2[:,0] = uW*(lam/(dx**2))
        resu2[:,-1] = uE*(lam/(dx**2))
        resu = resu1+resu2

        resv1 = np.zeros((m-1,n))
        resv2 = np.zeros((m-1,n))

        # North and South boundary
        resv2[0,:] = vN*(lam/(dy**2))
        resv2[-1,:] = vS*(lam/(dy**2))

        # West and East boundary
        resv1[:,0] = (16.0/5)*vW*(lam/(dx**2))
        resv1[:,-1] = (16.0/5)*vE*(lam/(dx**2))

        resv = resv1+resv2
        rhs_uvstarcd = rhs_uvstar + [resu, resv]

        return rhs_uvstarcd

class Alg3_method():
    '''This class constructs the Alg2 method (pressure free) solver'''

    def __init__(self, Re, mesh):
        self.Re = Re
        self.n = mesh.n
        self.m = mesh.m
        self.xu = mesh.xu
        self.yu = mesh.yu
        self.xv = mesh.xv
        self.yv = mesh.yv
        self.gds = mesh.gds
        self.sdomain = mesh.sdomain
        self.tdomain = mesh.tdomain
        self.Tn = mesh.Tn
        self.t0 = mesh.tdomain[0]
        self.dt = mesh.dt
        self.dx = mesh.dx
        self.dy = mesh.dy
        self.mesh = mesh

    # initial set up
    def setup(self, InCond_uv_init, Boundary_uv_type, solve_method='ILU', integration_method='Riemann'):
        ## InCond_uv: specifies the velocity initial condition 
        linsys_solver = LinearSystem_solver(self.Re, self.mesh)
        phi_mat = linsys_solver.Poisson_pressure_matrix(solve_method)
        u_mat = linsys_solver.Linsys_velocity_matrix("u")
        v_mat = linsys_solver.Linsys_velocity_matrix("v")

        InCond_uvcmp = structure.VelocityComplete(self.mesh, InCond_uv_init, 0).complete(Boundary_uv_type)
        uv_cmp = copy.copy(InCond_uvcmp)        
        initial_setup_parameters = [phi_mat, u_mat, v_mat, InCond_uvcmp, uv_cmp, integration_method, solve_method]
        return initial_setup_parameters

    def iterative_solver(self, Boundary_uv_type, Tn, initial_setup_parameters):
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt
        Re = self.Re
        phi_mat = initial_setup_parameters[0]
        u_mat = initial_setup_parameters[1]
        v_mat = initial_setup_parameters[2]
        # uvold_cmp: u and v velocity fields at time n-1
        # cmp: in the completed format (interior + boundary + ghost nodes)
        uvold_cmp = initial_setup_parameters[3]
        # uvn_cmp: u and v at time n
        uvn_cmp = initial_setup_parameters[4]
        integration_method = initial_setup_parameters[5] 
        solve_method = initial_setup_parameters[6] 
        # int: interior points only
        uvn_int = structure.VelocityField(uvn_cmp.get_int_uv()[0], uvn_cmp.get_int_uv()[1], self.mesh)
        # phiold: phi variable at time n-1
        phiold = np.zeros((m,n))
        phiold_cmp = structure.CentredPotential(phiold, self.mesh).complete()
        # phin_cmp: phi variable at time n
        phin_cmp = np.copy(phiold_cmp)

        print(Tn, "number of iterations")
        # main iterative solver
        test_problem_name = Boundary_uv_type
        for t in range(Tn):
            forcing_term = structure.Forcing_term(self.mesh,test_problem_name,t+0.5).select_forcing_term()
            convc_uv = uvn_cmp.non_linear_convection()
            preconvc_uv = uvold_cmp.non_linear_convection()
            diff_uvn = uvn_cmp.diffusion()
            if Boundary_uv_type == 'periodic_forcing_1':
                # Stokes problem
                rhs_uvstar = uvn_int + dt*((1.0/(2*Re))*diff_uvn + forcing_term)
            elif Boundary_uv_type == 'periodic_forcing_2':
                # Stokes problem
                rhs_uvstar = uvn_int + dt*((1.0/(2*Re))*diff_uvn + forcing_term)
            else:
                # full Navier Stokes problem
                rhs_uvstar = uvn_int + dt*(-1.5*convc_uv + 0.5*preconvc_uv + (1.0/(2*Re))*diff_uvn + forcing_term) 

            # calculate the approximation to phi at time n+1
            gradphiuv = self.gradphi_app(phiold_cmp, phin_cmp)
            # boundary correction step
            rhs_uvstarcd = self.correct_boundary(rhs_uvstar, t+1, Boundary_uv_type, gradphiuv)
            # solving for the intermediate velocity variable uv*
            Linsys_solve = LinearSystem_solver(Re, self.mesh)
            uvstar = Linsys_solve.Linsys_velocity_solver([u_mat,v_mat], rhs_uvstarcd)
            uvstarcmp = structure.VelocityComplete(self.mesh, [uvstar.get_uv()[0],  uvstar.get_uv()[1]], t+1).complete(Boundary_uv_type)
            div_uvstar = uvstarcmp.divergence()

            # solving for the phi variable
            phi = Linsys_solve.Poisson_pressure_solver(div_uvstar/dt, solve_method, phi_mat)
            # pressure correction step
            p = phi - div_uvstar/(2*Re)
            print(self.mesh.integrate(p, integration_method), 'integral of p')

            gradp = p.gradient()
            phiold_cmp = np.copy(phin_cmp)
            phin_cmp = np.copy(phi.complete())
            # velocity update stemp
            gradphi = phi.gradient()
            uvn_int = uvstar - dt*gradphi
            uvold_cmp = copy.copy(uvn_cmp)
            uvn_cmp = structure.VelocityComplete(self.mesh, [uvn_int.get_uv()[0],  uvn_int.get_uv()[1]], t+1).complete(Boundary_uv_type)
            print("iteration "+str(t))
            #break
        return uvn_cmp, p, gradp

    ## this function calculates graident of phi at time n+1
    # using second order approximation to gradient of phi^(n+1). Used in correcting uv*
    # phi^{n+1} appro 2*phi^n - phi^{n-1}
    def gradphi_app(self, phiold_cmp, phin_cmp):
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt

        phiapp_cmp = 2*phin_cmp - phiold_cmp
        gradphiu = (phiapp_cmp[:,1:n+2] - phiapp_cmp[:,0:n+1])/dx
        gradphiv = (phiapp_cmp[1:m+2,:] - phiapp_cmp[0:m+1,:])/dy
        # obtain gradphiu North and South boundary by cubic interpolation
        gradphiuN = 5.0/16*(gradphiu[0,:] +3*gradphiu[1,:] - gradphiu[2,:]+0.2*gradphiu[3,:])
        gradphiuS = 5.0/16*(gradphiu[-1,:] +3*gradphiu[-2,:] - gradphiu[-3,:]+0.2*gradphiu[-4,:])
        gradphiu[0,:] = gradphiuN
        gradphiu[-1,:] = gradphiuS

        # obtain gradphiv West and East boundary by cubic interpolation
        gradphivW = 5.0/16*(gradphiv[:,0] +3*gradphiv[:,1] - gradphiv[:,2]+0.2*gradphiv[:,3])
        gradphivE = 5.0/16*(gradphiv[:,-1] +3*gradphiv[:,-2] - gradphiv[:,-3]+0.2*gradphiv[:,-4])
        gradphiv[:,0] = gradphivW
        gradphiv[:,-1] = gradphivE
        return [gradphiu, gradphiv]

    # boundary correction used in solving for the intermediate velocity field (uv*)
    def correct_boundary(self, rhs_uvstar, t, Boundary_type, gradphiuv):
        # rhsuv is a VelocityField object with dimension interior u and v [(m*(n-1), (m-1)*n)]
        n = self.n
        m = self.m
        Re = self.Re
        dx = self.dx
        dy = self.dy
        dt = self.dt

        lam = dt/(2.0*Re)
        VC = structure.VelocityComplete(self.mesh, [rhs_uvstar.get_uv()[0], rhs_uvstar.get_uv()[1]], t)
        gradphiu = gradphiuv[0]
        gradphiv = gradphiuv[1]

        if Boundary_type == "driven_cavity":
            uN = VC.bnd_driven_cavity('u')['N']
            uS = VC.bnd_driven_cavity('u')['S']
            uW = VC.bnd_driven_cavity('u')['W']
            uE = VC.bnd_driven_cavity('u')['E']

            vN = VC.bnd_driven_cavity('v')['N']
            vS = VC.bnd_driven_cavity('v')['S']
            vW = VC.bnd_driven_cavity('v')['W']
            vE = VC.bnd_driven_cavity('v')['E']

        elif Boundary_type == "Taylor":
            uN = VC.bnd_Taylor('u')['N'][1:n]
            uS = VC.bnd_Taylor('u')['S'][1:n]
            uW = VC.bnd_Taylor('u')['W']
            uE = VC.bnd_Taylor('u')['E']

            vN = VC.bnd_Taylor('v')['N']
            vS = VC.bnd_Taylor('v')['S']
            vW = VC.bnd_Taylor('v')['W'][1:m]
            vE = VC.bnd_Taylor('v')['E'][1:m]

        elif Boundary_type == "periodic_forcing_1":
            uN = VC.bnd_forcing_1('u')['N'][1:n]
            uS = VC.bnd_forcing_1('u')['S'][1:n]
            uW = VC.bnd_forcing_1('u')['W']
            uE = VC.bnd_forcing_1('u')['E']

            vN = VC.bnd_forcing_1('v')['N']
            vS = VC.bnd_forcing_1('v')['S']
            vW = VC.bnd_forcing_1('v')['W'][1:m]
            vE = VC.bnd_forcing_1('v')['E'][1:m]

        elif Boundary_type == "periodic_forcing_2":
            uN = VC.bnd_forcing_2('u')['N'][1:n]
            uS = VC.bnd_forcing_2('u')['S'][1:n]
            uW = VC.bnd_forcing_2('u')['W']
            uE = VC.bnd_forcing_2('u')['E']

            vN = VC.bnd_forcing_2('v')['N']
            vS = VC.bnd_forcing_2('v')['S']
            vW = VC.bnd_forcing_2('v')['W'][1:m]
            vE = VC.bnd_forcing_2('v')['E'][1:m]

        gradphiuW = gradphiu[1:m+1,0]
        gradphiuE = gradphiu[1:m+1,-1]
        gradphiuN = gradphiu[0,1:n]
        gradphiuS = gradphiu[-1,1:n]

        # North and South boundary
        uNbc = uN + dt*gradphiuN
        uSbc = uS + dt*gradphiuS

        resu1 = np.zeros((m,n-1))
        resu2 = np.zeros((m,n-1))
        resu1[0,:] = (16.0/5)*(uNbc)*(lam/(dy**2))
        resu1[-1,:] = (16.0/5)*(uSbc)*(lam/(dy**2))            

        # West and East boundary
        uWbc = uW
        uEbc = uE
        resu2[:,0] = (uWbc)*(lam/(dx**2))
        resu2[:,-1] = (uEbc)*(lam/(dx**2))
        resu = resu1+resu2

        resv1 = np.zeros((m-1,n))
        resv2 = np.zeros((m-1,n))

        gradphivN = gradphiv[0,1:n+1]
        gradphivS = gradphiv[-1,1:n+1]
        gradphivW = gradphiv[1:m,0]
        gradphivE = gradphiv[1:m,-1]

        # North and South boundary
        vNbc = vN
        vSbc = vS
        resv2[0,:] = vNbc*(lam/(dy**2))
        resv2[-1,:] = vSbc*(lam/(dy**2))

        # West and East boundary
        vWbc = vW + dt*gradphivW
        vEbc = vE + dt*gradphivE
        resv1[:,0] = (16.0/5)*vWbc*(lam/(dx**2))
        resv1[:,-1] = (16.0/5)*vEbc*(lam/(dx**2))

        resv = resv1+resv2
        rhs_uvstarcd = rhs_uvstar + [resu, resv]

        return rhs_uvstarcd

class Error():
    ''' This class calculates the error norms for the solver by comparing the numerical and analyticalsolutions'''

    def __init__(self, uv_cmp, uv_exact_bnd, p, p_exact, gradp, gradp_exact, div_uv, mesh):
        self.mesh = mesh
        self.uv_cmp = uv_cmp
        self.uv_bnd = uv_cmp.get_bnd_uv()
        self.uv_exact_bnd = uv_exact_bnd
        self.p_exact = p_exact
        self.p = p
        self.gradp = gradp
        self.gradp_exact = gradp_exact
        self.div_uv = div_uv

    def velocity_error(self):
        n = self.mesh.n
        m = self.mesh.m
        # m: row, n: col
        uebnd = self.uv_bnd[0] - self.uv_exact_bnd.get_uv()[0]
        vebnd = self.uv_bnd[1] - self.uv_exact_bnd.get_uv()[1]
        L1 = []
        L2 = []
        Linf = []

        for x in [uebnd, vebnd]:
            xv = np.ravel(x)
            a=sum(abs(xv[:])**2)/(m**2)
#	    L2x = np.sqrt(sum(xv[:]**2))/(m**2)
            Linfx = abs(xv[:]).max()
            L1x = sum(abs(xv[:]))/(m**2)
            L1.append(L1x)
            L2x = np.sqrt(a)
            L2.append(L2x)
            Linf.append(Linfx)
        ubnderror = {'L1': L1[0], 'L2': L2[0], 'Linf': Linf[0]}
        vbnderror = {'L1': L1[1], 'L2': L2[1], 'Linf': Linf[1]}

        return ubnderror, vbnderror

    def pressure_error(self):
        n = self.mesh.n
        m = self.mesh.m

        perror = self.p - self.p_exact
        pv = np.ravel(perror.get_value())
        a=sum(abs(pv[:])**2)/(m**2)
#	L2p = np.sqrt(sum(pv[:]**2))/(m**2)
        Linfp = abs(pv[:]).max()
        L1p = sum(abs(pv[:]))/(m**2)
        L2p = np.sqrt(a)
        perror_dict = {'L1': L1p, 'L2': L2p, 'Linf': Linfp}

        return perror_dict

    def pressure_gradient_error(self):
        n = self.mesh.n
        m = self.mesh.m

        gradp_error = self.gradp - self.gradp_exact
        gradpu_error, gradpv_error = gradp_error.get_uv()
        gradpu_errorv = np.ravel(gradpu_error)
        gradpv_errorv = np.ravel(gradpv_error)
        gradperror_list = []
        for gradpe in [gradpu_errorv, gradpv_errorv]:
            a=sum(abs(gradpe[:])**2)/(m**2)
            Linfp = abs(gradpe[:]).max()
            L1p = sum(abs(gradpe[:]))/(m**2)
            L2p = np.sqrt(a)
            gradperror_dict = {'L1': L1p, 'L2': L2p, 'Linf': Linfp}
            gradperror_list.append(gradperror_dict)
        avg_gradp_error_dict = {'L1': (gradperror_list[0]['L1']+gradperror_list[1]['L1'])/2, 'L2': (gradperror_list[0]['L2']+gradperror_list[1]['L2'])/2, 'Linf': (gradperror_list[0]['Linf']+gradperror_list[1]['Linf'])/2}

        return gradperror_list[0], gradperror_list[1], avg_gradp_error_dict


