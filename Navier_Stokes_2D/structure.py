# -*- coding: utf-8 -*-
"""
This file sets up the basic structure of the numerical solver, including
the spatial geometry and the structure of velocity and potential fields.
"""

from __future__ import division
import numpy as np
import time

# 2D Navier Stokes solver
# mesh contains the meshgrid for velocity and pressure
# gridsize = [m,n]
# spatial_domain = [[xl, xr], [yl,yr]]
# time_domain = [t0, tend]

__all__ = ['mesh', 'VelocityField', 'VelocityComplete', 
	'InitialCondition', 'CentredPotential', 'Exact_solutions']
 
class mesh:
    '''This class constructurs the structure of meshgrids for velocity and pressure'''
    def __init__(self, gridsize, spatial_domain, time_domain, CFL, Re):
        # m: row, n: column
        self.gds = gridsize
        self.m = gridsize[0]
        self.n = gridsize[1]
        self.sdomain = spatial_domain
        self.tdomain = time_domain
        self.CFL = CFL
        # dx, dy: delta x and delta y
        self.dx = abs(float(self.sdomain[0][1] - self.sdomain[0][0]))/self.n
        self.dy = abs(float(self.sdomain[1][1] - self.sdomain[1][0]))/self.m
        # dt: delta t
        #        self.dt1 = abs(((self.sdomain[0][1] - self.sdomain[0][0])/self.gds[0])*CFL)
        self.dt1 = CFL/(1.0/self.dx + 1.0/self.dy)
        # tn: number of iterations
        self.Tn = int(round(self.tdomain[1]/self.dt1))
        self.dt = abs(float(self.tdomain[1] - self.tdomain[0]))/self.Tn
        # xu, yu: horizontal velocity grids
        # xv, yvv: vertical velocity grids
        self.xu = np.linspace(start=self.sdomain[0][0], stop=self.sdomain[0][1],num=self.n+1)
        self.yu = np.linspace(start=self.sdomain[1][0]+0.5*self.dy, stop=self.sdomain[1][1]-0.5*self.dy,num=self.m)
        self.xv = np.linspace(start=self.sdomain[0][0]+0.5*self.dx, stop=self.sdomain[0][1]-0.5*self.dx,num=self.n)
        self.yv = np.linspace(start=self.sdomain[1][0], stop=self.sdomain[1][1],num=self.m+1)
        self.Re = Re

    # functions ubndmg, vbndmg, uintmg, vintmg and pintmg returns the meshgrids for velocities and pressure
    # bnd: grid including boundary points; int: grid only containing interior points
    def ubndmg(self, x):
        # mg means mesh grid
        # Xu and Yu include boundary locations
        Xu, Yu = np.meshgrid(self.xu,self.yu)
        # u is a list of x and y grids
        if x == "x":
            return Xu
        else:
            return Yu

    def vbndmg(self, x):
        Xv, Yv = np.meshgrid(self.xv,self.yv)
        if x == "x":
            return Xv
        else:
            return Yv

    def uintmg(self, x):
        xuint = self.xu[1:-1]
        Xuint, Yuint = np.meshgrid(xuint, self.yu)
        if x == "x":
            return Xuint
        else:
            return Yuint

    def vintmg(self, x):
        yvint = self.yv[1:-1]
        Xvint, Yvint = np.meshgrid(self.xv, yvint)
        if x == "x":
            return Xvint
        else:
            return Yvint

    def pintmg(self, x):
        XPint, YPint = np.meshgrid(self.xv, self.yu)
        if x == "x":
            return XPint
        else:
            return YPint

    # this function integrates a CentredPotential object
    # this is mainly used in solving the Poisson pressure linear system
    def integrate(self, p_int=None, integration_method='Riemann'):
        n = self.n
        m = self.m
        dh = self.dx

        if integration_method == 'Riemann':
            # use Riemann sum to approximate the integral
            h = dh**2
            C1 = np.ones(n*m)
            C = h*C1
        # other methods such as Simpson's rule could be explored
        else:
            raise TypeError('only Riemann sum is implemented')

        if p_int == None:
            # return the integration matrix as a 1 D array
            return C
        else:
            if p_int.__class__.__name__ != 'CentredPotential':
                raise TypeError('only CentredPotential objects can be integrated by this function')
            else:
                intg_p = np.dot(C, p_int.get_value().ravel())
                return intg_p

# This class constructs the structure of velocity fields (u, v)     
# It defines the basic operations and properties of velocity fields
class VelocityField:
    def __init__(self, ucmp, vcmp, mesh):
        
        # the class assumes u and v are in the complete form: interior + boundary + ghost nodes
        self.ucmp = ucmp
        self.vcmp = vcmp
        self.mesh = mesh

    # returns the complete points (interior + boundary + ghost nodes) for u and v in the form of numpy arraies 
    def get_uv(self):
        return [self.ucmp, self.vcmp]
    # returns the interior points of u and v in the form of numpy arries
    def get_int_uv(self):
        n = self.mesh.n
        m = self.mesh.m
        return [self.ucmp[1:m+1,1:n], self.vcmp[1:m,1:n+1]]
    # returns the interior and boundary points of u and v in the form of numpy arries
    def get_bnd_uv(self):
        n = self.mesh.n
        m = self.mesh.m        
        u_bnd = self.ucmp[1:m+1,0:n+1]
        v_bnd = self.vcmp[0:m+1,1:n+1]        
        return [u_bnd, v_bnd]
    
    # defines the basic operations for velocity fields
    def __neg__(self):
        # negation
        return VelocityField(-self.ucmp, -self.vcmp, self.mesh)
        
    def __add__(self, other):
        ## addition between two VelocityField instances
        try:
            ou, ov = other.get_uv()
            nu = self.ucmp + ou
            nv = self.vcmp + ov
        ## addition with numpy arraies or list of numpy arraies
        except AttributeError:
            try:
                nu = self.ucmp + other[0]
                nv = self.vcmp + other[1]
            ## addition with integer
            ## assume the integer is added to both horizontal and vertical velocities
            except TypeError:
                nu = self.ucmp + other
                nv = self.vcmp + other                
        return VelocityField(nu, nv, self.mesh)

    def __radd__(self, other):
        ## right addition between two VelocityField instances
        #print other, "other radd"
        return self.__rsub__(-other)

    def __sub__(self, other):
        return self.__add__(-other)
        
    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        # multiplication only defined between VelocityField instances and integers
        nu = self.ucmp*other
        nv = self.vcmp*other
        return VelocityField(nu, nv, self.mesh) 
        
    def __rmul__(self, other):
        nu = self.ucmp*other
        nv = self.vcmp*other
        return VelocityField(nu, nv, self.mesh)

    def __truediv__(self, other):
        # division only defined between VelocityField instances and integers
        return self.__mul__(1.0/other)

    ## below defines the divergence, difussion and non linear convection operations
    def divergence(self):
        # calculate the divergence of (u,v) at interior points
        # uv_cmp must be completed with boundary and ghose points. Dimension: m+2 x n+1, m+1 x n+2  
        m = self.mesh.m
        n = self.mesh.n
        ubnd = self.ucmp[1:m+1,:]
        vbnd = self.vcmp[:,1:n+1]
        dx = self.mesh.dx
        dy = self.mesh.dy
               
        div = (ubnd[:,1:n+1] - ubnd[:,0:n])/dx +\
              (vbnd[1:m+1,:] - vbnd[0:m,:])/dy
        divPotentialField = CentredPotential(div, self.mesh)
        
        return divPotentialField
        
    def diffusion(self):
        # calculate the diffusive terms of u (v) at interior points
        # uv_cmp must be completed with boundary and ghose points. Dimension: m+2 x n+1, m+1 x n+2
        n = self.mesh.n
        m = self.mesh.m
        dx = self.mesh.dx
        dy = self.mesh.dy
        
        u = self.ucmp
        v = self.vcmp
        
        diffu = (u[1:m+1,2:n+1] -2*u[1:m+1,1:n] + u[1:m+1,0:n-1])/(dx**2) +\
                (u[2:m+2,1:n] - 2*u[1:m+1,1:n] + u[0:m,1:n])/(dy**2)
        diffv = (v[1:m,2:n+2] - 2*v[1:m,1:n+1] + v[1:m,0:n])/(dx**2) +\
                (v[2:m+1,1:n+1] - 2*v[1:m,1:n+1] + v[0:m-1,1:n+1])/(dy**2)
        
        return VelocityField(diffu, diffv, self.mesh)
    
    def non_linear_convection(self):
        # calculate the convective terms of u (v) at interior points
        # use 4 point average to calculate u and v values at pressure nodes
        # uv_cmp must be completed with boundary and ghost points m+2 x n+1, m+1 x n+2
        n = self.mesh.n
        m = self.mesh.m
        dx = self.mesh.dx
        dy = self.mesh.dy

        u = self.ucmp
        v = self.vcmp

        # average U and V (4 point average)
        uah = 0.5*(u[:,1:n+1] + u[:,0:n])
        ua = 0.5*(uah[2:m+1,:] + uah[1:m,:])
        vah = 0.5*(v[:,2:n+1] + v[:,1:n])
        va = 0.5*(vah[1:m+1,:] + vah[0:m,:])
        
        convcu = u[1:m+1,1:n]*(u[1:m+1,2:n+1] - u[1:m+1,0:n-1])/(2*dx) +\
                 va[:]*(u[2:m+2,1:n] - u[0:m,1:n])/(2*dy)
        convcv = ua[:]*(v[1:m,2:n+2] - v[1:m,0:n])/(2*dx) +\
                 v[1:m,1:n+1]*(v[2:m+1,1:n+1] - v[0:m-1,1:n+1])/(2*dy)
        return VelocityField(convcu, convcv, self.mesh)

class VelocityComplete:
    '''This class complete the velocity fields (i.e adding boundary and ghost points)
       mesh is the mesh class, uv_int=[u_int, v_int] is a list of interior u and v in the form of numpy arries
       t: the t th iteration'''

    def __init__(self, mesh, uv_int, t):
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
        self.uv_int = uv_int
        self.t = t
        self.mesh = mesh

    # returns the boundary points for the Driven cavity flow problem
    def bnd_driven_cavity(self, u):
        if u == "u":
            # returns the boundary value for u nodes: all 4 sides (South, East, West and North) included
            # using the lid driven condition            
            ubnd_value = {'S': 0, 'E': 0, 'W': 0, 'N': 0}
            return ubnd_value
        elif u == "v":
            # returns the v boundary points along all 4 sides
	    # currently the East boundary of V velocity is set to be constant 1 (this is flexible)
            vbnd_value = {'S': 0, 'E': 1, 'W': 0, 'N': 0}
            return vbnd_value
            
    # returns the boundary points for unforced Taylor flow full Navier Stokes problem
    def bnd_Taylor(self, u):
        tn = self.dt*self.t + self.t0
        if u == "u":
            uN = -np.cos(self.xu)*np.sin(self.sdomain[1][0])*np.exp(-2*tn)
            uS = -np.cos(self.xu)*np.sin(self.sdomain[1][1])*np.exp(-2*tn)
            uW = -np.cos(self.sdomain[0][0])*np.sin(self.yu)*np.exp(-2*tn)
            uE = -np.cos(self.sdomain[0][1])*np.sin(self.yu)*np.exp(-2*tn)
            ubnd_value = {'S': uS, 'E': uE, 'W': uW, 'N': uN}            
            return ubnd_value
            
        elif u == "v":
            vN = np.sin(self.xv)*np.cos(self.sdomain[1][0])*np.exp(-2*tn)
            vS = np.sin(self.xv)*np.cos(self.sdomain[1][1])*np.exp(-2*tn)
            vW = np.sin(self.sdomain[0][0])*np.cos(self.yv)*np.exp(-2*tn)
            vE = np.sin(self.sdomain[0][1])*np.cos(self.yv)*np.exp(-2*tn)
            vbnd_value = {'S': vS, 'E': vE, 'W': vW, 'N': vN}
            return vbnd_value

    # returns the boundary points for the first type of forced flow linearised Navier Stokes problem
    def bnd_forcing_1(self, u):
        tn = self.dt*self.t + self.t0
        xu, yu = self.xu, self.yu
        xv, yv = self.xv, self.yv
        if u == "u":
            uN = np.pi*np.sin(tn)*np.sin(2*np.pi*self.sdomain[1][0])*(np.sin(np.pi*xu)**2)
            uS = np.pi*np.sin(tn)*np.sin(2*np.pi*self.sdomain[1][1])*(np.sin(np.pi*xu)**2)
            uW = np.pi*np.sin(tn)*np.sin(2*np.pi*yu)*(np.sin(np.pi*self.sdomain[0][0])**2)
            uE = np.pi*np.sin(tn)*np.sin(2*np.pi*yu)*(np.sin(np.pi*self.sdomain[0][1])**2)
            ubnd_value = {'S': uS, 'E': uE, 'W': uW, 'N': uN}
            return ubnd_value
        
        elif u == "v":
            vN = -np.pi*np.sin(tn)*np.sin(2*np.pi*xv)*(np.sin(np.pi*self.sdomain[1][0])**2)
            vS = -np.pi*np.sin(tn)*np.sin(2*np.pi*xv)*(np.sin(np.pi*self.sdomain[1][1])**2)
            vW = -np.pi*np.sin(tn)*np.sin(2*np.pi*self.sdomain[0][0])*(np.sin(np.pi*yv)**2)
            vE = -np.pi*np.sin(tn)*np.sin(2*np.pi*self.sdomain[0][1])*(np.sin(np.pi*yv)**2)
            vbnd_value = {'S': vS, 'E': vE, 'W': vW, 'N': vN}
            return vbnd_value

    # returns the boundary points for the second type of forced flow linearised Navier Stokes problem
    def bnd_forcing_2(self, u):
        tn = self.dt*self.t + self.t0
        xu, yu = self.xu, self.yu
        xv, yv = self.xv, self.yv
        if u == "u":
            uN = np.sin(xu + tn)*np.sin(self.sdomain[1][0] + tn)
            uS = np.sin(xu + tn)*np.sin(self.sdomain[1][1] + tn)
            uW = np.sin(self.sdomain[0][0] + tn)*np.sin(yu + tn)
            uE = np.sin(self.sdomain[0][1] + tn)*np.sin(yu + tn)
            ubnd_value = {'S': uS, 'E': uE, 'W': uW, 'N': uN}
            return ubnd_value
        
        elif u == "v":
            vN = np.cos(xv + tn)*np.cos(self.sdomain[1][0] + tn)
            vS = np.cos(xv + tn)*np.cos(self.sdomain[1][1] + tn)
            vW = np.cos(self.sdomain[0][0] + tn)*np.cos(yv + tn)
            vE = np.cos(self.sdomain[0][1] + tn)*np.cos(yv + tn)
            vbnd_value = {'S': vS, 'E': vE, 'W': vW, 'N': vN}
            return vbnd_value      
    
    # this function completes (add boundary and ghost points) the u and v velocity fields 
    def complete(self, Boundary_type, return_bnd=False):
        # u and v only given interior points m x n-1, m-1 x n
        n = self.n
        m = self.m
        
        if Boundary_type == "driven_cavity":
            uN = self.bnd_driven_cavity('u')['N']
            uS = self.bnd_driven_cavity('u')['S']
            uW = self.bnd_driven_cavity('u')['W']
            uE = self.bnd_driven_cavity('u')['E']
        
            vN = self.bnd_driven_cavity('v')['N']
            vS = self.bnd_driven_cavity('v')['S']
            vW = self.bnd_driven_cavity('v')['W']
            vE = self.bnd_driven_cavity('v')['E']

        elif Boundary_type == "Taylor":
            uN = self.bnd_Taylor('u')['N']
            uS = self.bnd_Taylor('u')['S']
            uW = self.bnd_Taylor('u')['W']
            uE = self.bnd_Taylor('u')['E']
        
            vN = self.bnd_Taylor('v')['N']
            vS = self.bnd_Taylor('v')['S']
            vW = self.bnd_Taylor('v')['W']
            vE = self.bnd_Taylor('v')['E']

        elif Boundary_type == "periodic_forcing_1":
            uN = self.bnd_forcing_1('u')['N']
            uS = self.bnd_forcing_1('u')['S']
            uW = self.bnd_forcing_1('u')['W']
            uE = self.bnd_forcing_1('u')['E']
        
            vN = self.bnd_forcing_1('v')['N']
            vS = self.bnd_forcing_1('v')['S']
            vW = self.bnd_forcing_1('v')['W']
            vE = self.bnd_forcing_1('v')['E']

        elif Boundary_type == "periodic_forcing_2":
            uN = self.bnd_forcing_2('u')['N']
            uS = self.bnd_forcing_2('u')['S']
            uW = self.bnd_forcing_2('u')['W']
            uE = self.bnd_forcing_2('u')['E']
        
            vN = self.bnd_forcing_2('v')['N']
            vS = self.bnd_forcing_2('v')['S']
            vW = self.bnd_forcing_2('v')['W']
            vE = self.bnd_forcing_2('v')['E']

        u = np.zeros((m+2,n+1))
        v = np.zeros((m+1,n+2))
        u[1:m+1,1:n] = self.uv_int[0]
        v[1:m,1:n+1] = self.uv_int[1]
        
        # add boundary and ghost points in
        # ghost nodes added using cubic polynomial interpolation
        
        # for the West and East u boundaries
        u[1:m+1,0] = uW
        u[1:m+1,-1] = uE
        # for the North and South u boundaries
        # cubic interpolation
        u[0,:] = (16.0/5)*uN - 3*u[1,:] + u[2,:] - (1.0/5)*u[3,:]
        u[-1,:] = (16.0/5)*uS - 3*u[-2,:] + u[-3,:] - (1.0/5)*u[-4,:]
        
        # for the North and south v boundaries
        v[0,1:n+1] = vN
        v[-1,1:n+1] = vS
        # for the West and East v boundaries
        # cubic interpolation
        v[:,0] = (16.0/5)*vW - 3*v[:,1] + v[:,2] - (1.0/5)*v[:,3]
        v[:,-1] = (16.0/5)*vE - 3*v[:,-2] + v[:,-3] - (1.0/5)*v[:,-4]
        if return_bnd == False:
            # if only want the VelocityField instance
            return VelocityField(u, v, self.mesh)
        else:
            # if wanat the boundary points for u and v in numpy array format
            return VelocityField(u, v, self.mesh), [[uN, uS, uW, uE], [vN, vS, vW, vE]]

class InitialCondition:
    '''This class contains the set of inition conditions (could be extended later) 
       the returned objects are u and v as well as pressure in the form of numpy arrays with only interior points.
       It can be turned into VelocityField or CentredPotential instances using the VelocityComplete or CentredPotential classes'''

    def __init__(self, mesh, integration_method='Riemann'):
        self.mesh = mesh
        self.n = mesh.n
        self.m = mesh.m
        self.xu = mesh.xu
        self.yu = mesh.yu
        self.xv = mesh.xv
        self.yv = mesh.yv
        self.Xuint = mesh.uintmg("x")
        self.Yuint = mesh.uintmg("y")
        self.Xvint = mesh.vintmg("x")
        self.Yvint = mesh.vintmg("y")
        self.XPint = mesh.pintmg("x")
        self.YPint = mesh.pintmg("y")
        self.Re = mesh.Re
        self.integration_method = integration_method

    # zero inition condition for velocity fields
    def zero_uv(self):
        n = self.n
        m = self.m
        # n, m defines the size of grid
        # this function constructs the interior points of u and b
        # u velocity has dimension m x n+1 and v has dimension m+1 x n
        # u interior has m x n-1, v has interior m - 1 x n
        # with ghost nodes: u: m+2 x n+1, v: m+1 x n+2
        gridu = np.zeros((m,n-1))
        gridv = np.zeros((m-1,n))
        grid_int = [gridu, gridv]
        ic_int = grid_int
        return ic_int

    # Taylor flow problem
    def Taylor_uv(self):
        Xuint, Yuint = self.Xuint, self.Yuint
        Xvint, Yvint = self.Xvint, self.Yvint

        U = -np.cos(Xuint)*np.sin(Yuint)
        V = np.sin(Xvint)*np.cos(Yvint)
        ic = [U,V]
        return ic

    def periodic_forcing_2_uv(self):
        Xuint, Yuint = self.Xuint, self.Yuint
        Xvint, Yvint = self.Xvint, self.Yvint

        U = np.sin(Xuint)*np.sin(Yuint)
        V = np.cos(Xvint)*np.cos(Yvint)
        ic = [U,V]
        return ic

    # zero initial condition for pressure
    def zero_P(self):
        # n, m defines the size of grid
        # p only has interior points (m x n)
        n = self.n
        m = self.m
        gridP = np.zeros((m,n))
        icP = gridP
        return icP

    # Taylor flow problem
    def Taylor_P(self):
        n = self.n
        m = self.m
        XPint, YPint = self.XPint, self.YPint
        P = -self.Re*0.25*(np.cos(2*XPint) + np.cos(2*YPint))
        # normalise P so that it satisfies the zero integral constraint 
        CP = self.mesh.integrate(CentredPotential(P, self.mesh), self.integration_method)
        C1 = self.mesh.integrate(CentredPotential(np.ones((n,m)), self.mesh), self.integration_method)
        c = CP/C1
        c = np.sum(P)/(n*m)
        P = P - c

        return P

    def periodic_forcing_2_P(self):
        n = self.n
        m = self.m
        XPint, YPint = self.XPint, self.YPint
        P = (1.0/self.Re)*np.sin(XPint-YPint)
        # normalise P so that it satisfies the zero integral constraint
        CP = self.mesh.integrate(CentredPotential(P, self.mesh), self.integration_method)
        C1 = self.mesh.integrate(CentredPotential(np.ones((n,m)), self.mesh), self.integration_method)
        c = CP/C1
        c = np.sum(P)/(n*m)
        P = P - c
        return P

    def select_initial_conditions(self, test_problem_name):
        if test_problem_name == 'Taylor':
            Init_uv = self.Taylor_uv()
            Init_P = self.Taylor_P()

        elif test_problem_name == 'periodic_forcing_1':
            Init_uv = self.zero_uv()
            Init_P = self.zero_P()

        elif test_problem_name == 'periodic_forcing_2':
            Init_uv = self.periodic_forcing_2_uv()
            Init_P = self.periodic_forcing_2_P()

        elif test_problem_name == 'driven_cavity':
            Init_uv = self.zero_uv()
            Init_P = self.zero_P()

        return Init_uv, Init_P
	    
class CentredPotential():
    '''This class constructs the structure of CentredPotential (pressure, scalar potential) objects
       It definies the basic operations and properties for CentredPotential objects'''

    def __init__(self, p_int, mesh):
        self.mesh = mesh
        self.p_int = p_int
    
    # complete function returns the pressure with ghost nodes (used only in solving phi field)
    # uses Neumann boundary condition
    # it returns a numpy array not CentredPotential object    
    def complete(self):
        n = self.mesh.n
        m = self.mesh.m
        p_cmp = np.zeros((m+2,n+2))
        p_cmp[1:m+1,1:n+1] = self.p_int
        # update ghost nodes
        # South
        p_cmp[-1,:] = p_cmp[-2,:]
        # North
        p_cmp[0,:] = p_cmp[1,:]
        # West
        p_cmp[:,0] = p_cmp[:,1]
        # East
        p_cmp[:,-1] = p_cmp[:,-2]
        
        return p_cmp
    
    # returns the interior points of pressure
    def get_value(self):
        return self.p_int

    # below defines the basic operations for CentredPotential objects
    def __neg__(self):
        return CentredPotential(-self.p_int, self.mesh)
        
    def __add__(self, other):
        ## addition between two CentredPotential instances
        try:
            op = other.get_value()
            np = self.p_int + op
        ## addition with numpy arraies or addition with integer
        except AttributeError:
            np = self.p_int + other        
            
        return CentredPotential(np, self.mesh)

    def __radd__(self, other):
        ## addition between two CentredPotential instances
        return self.__rsub__(-other)

    def __sub__(self, other):
        return self.__add__(-other)
        
    def __rsub__(self, other):
        return -self.__sub__(other)

    def __mul__(self, other):
        pn = self.p_int*other
        return CentredPotential(pn, self.mesh) 
        
    def __rmul__(self, other):
        pn = self.p_int*other
        return CentredPotential(pn, self.mesh)   

    def __truediv__(self, other):
        return self.__mul__(1.0/other)
        
    # defines the indexing for CentredPotential objects, return the resutls in the form of numpy array objects
    def __getitem__(self, key):
        return self.p_int.__getitem__(key)
    
    # calculate the Laplacian and Gradient
    def laplace(self):
        p = self.complete(self)
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        Lap_P = (p[1:m+1,2:n+2] -2*p[1:m+1,1:n+1] + p[1:m+1,0:n])/(dx**2) +\
                (p[2:m+2,1:n+1] - 2*p[1:m+1,1:n+1] + p[0:m,1:n+1])/(dy**2)
                
        return CentredPotential(Lap_P, self.mesh)
        
    def gradient(self):
        n = self.mesh.n
        m = self.mesh.m
        dx = self.mesh.dx
        dy = self.mesh.dy
        p = self.p_int

        px = (p[:,1:n] - p[:,0:n-1])/dx
        py = (p[1:m,:] - p[0:m-1,:])/dy
        return VelocityField(px, py, self.mesh)

class Exact_solutions():
    '''This class contains the exact solutions which are used in error analysis and comparisons with the numerical solutions
       It only contains unforced Talyor flow solutions, two periodic forcing flow solutions at the moment, others can be added later
       The returned exact solutions contain interior and boundary points
       The velocity fields are evaluated (in time) at integer indices (e.g k = 1, 2, 3...) whereas 
       pressure fields are evaluated (in time) at half integer indices (e.g k = 1/2, 3/2, 5/2 ...)'''

    def __init__(self, mesh, Re, t, integration_method='Riemann'):
        self.mesh = mesh
        self.t = t
        self.Re = Re
        self.Xubnd = mesh.ubndmg("x")
        self.Yubnd = mesh.ubndmg("y")
        self.Xvbnd = mesh.vbndmg("x")
        self.Yvbnd = mesh.vbndmg("y")
        self.XPint = mesh.pintmg("x")
        self.YPint = mesh.pintmg("y")
        self.Xuint = mesh.uintmg("x")
        self.Yuint = mesh.uintmg("y")
        self.Xvint = mesh.vintmg("x")
        self.Yvint = mesh.vintmg("y")
        self.n = mesh.n
        self.m = mesh.m
        self.integration_method = integration_method

    def Exact_solutions(self, Solution_type):
        dt = self.mesh.dt
        t = self.t
        Re = self.Re
        Xubnd, Yubnd = self.Xubnd, self.Yubnd
        Xvbnd, Yvbnd = self.Xvbnd, self.Yvbnd
        XPint, YPint = self.XPint, self.YPint
        Xuint, Yuint = self.Xuint, self.Yuint
        Xvint, Yvint = self.Xvint, self.Yvint
        n = self.n
        m = self.m

        tn = dt*t + self.mesh.tdomain[0]
        tnhalf = dt*(t-0.5) + self.mesh.tdomain[0]

        if Solution_type == "Taylor":
            U_exact_bnd = -np.cos(Xubnd)*np.sin(Yubnd)*np.exp(-2*tn)
            V_exact_bnd = np.sin(Xvbnd)*np.cos(Yvbnd)*np.exp(-2*tn)
            P_exact = -(1/(4*Re))*(np.cos(2*XPint) + np.cos(2*YPint))*np.exp(-4*tnhalf)
            # normalise P_exact, so that it satisfies the zero integral constraint
            CP = self.mesh.integrate(CentredPotential(P_exact, self.mesh), self.integration_method)
            C1 = self.mesh.integrate(CentredPotential(np.ones((n,m)), self.mesh), self.integration_method)
            c = CP/C1
            P_exact = P_exact - c
            gradpu_exact = (1/(2*Re))*np.sin(2*Xuint)*np.exp(-4*tnhalf)
            gradpv_exact = (1/(2*Re))*np.sin(2*Yvint)*np.exp(-4*tnhalf)

        elif Solution_type == 'periodic_forcing_1':
            U_exact_bnd = np.pi*np.sin(tn)*np.sin(2*np.pi*Yubnd)*(np.sin(np.pi*Xubnd)**2)
            V_exact_bnd = -np.pi*np.sin(tn)*np.sin(2*np.pi*Xvbnd)*(np.sin(np.pi*Yvbnd)**2)
            P_exact = np.sin(tnhalf)*np.sin(np.pi*YPint)*np.cos(np.pi*XPint)
            # normalise P_exact, so that it satisfies the zero integral constraint
            CP = self.mesh.integrate(CentredPotential(P_exact, self.mesh), self.integration_method)
            C1 = self.mesh.integrate(CentredPotential(np.ones((n,m)), self.mesh), self.integration_method)
            c = CP/C1
            P_exact = P_exact - c
            gradpu_exact = -np.pi*np.sin(tnhalf)*np.sin(np.pi*Xuint)*np.sin(np.pi*Yuint)
            gradpv_exact = np.pi*np.sin(tnhalf)*np.cos(np.pi*Xvint)*np.cos(np.pi*Yvint)

        elif Solution_type == 'periodic_forcing_2':
            U_exact_bnd = np.sin(Xubnd+tn)*np.sin(Yubnd+tn)
            V_exact_bnd = np.cos(Xvbnd+tn)*np.cos(Yvbnd+tn)
            P_exact = np.sin(XPint-YPint+tnhalf)
            # normalise P_exact, so that it satisfies the zero integral constraint
            CP = self.mesh.integrate(CentredPotential(P_exact, self.mesh), self.integration_method)
            C1 = self.mesh.integrate(CentredPotential(np.ones((n,m)), self.mesh), self.integration_method)
            c = CP/C1
            P_exact = P_exact - c
            gradpu_exact = np.cos(Xuint-Yuint+tnhalf)
            gradpv_exact = -np.cos(Xvint-Yvint+tnhalf)

        return VelocityField(U_exact_bnd, V_exact_bnd, self.mesh), CentredPotential(P_exact, self.mesh), VelocityField(gradpu_exact, gradpv_exact, self.mesh)
 
class Forcing_term:
    '''This class contains the external forcing term that are required for some flow problems e.g periodic_forcing_1'''

    def __init__(self, mesh, test_problem_name, t):
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
        self.t = t
        self.test_problem_name = test_problem_name
        self.Re = mesh.Re

    def periodic_forcing_1(self):
        # defines the first type of periodic forcing problem
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt
        t = self.t
        Re = self.Re
        # change all Xu and Xv, Yu, Yv into xu, xv, yu, yv
        Xu, Yu = np.meshgrid(self.xu, self.yu)
        Xv, Yv = np.meshgrid(self.xv, self.yv)

        tn = dt*t + self.tdomain[0]
        # these forcing terms include boundary points
        Fx = np.pi*np.cos(tn)*np.sin(2*np.pi*Yu)*(np.sin(np.pi*Xu)**2) -\
             2*(np.pi**3)*np.sin(tn)*np.sin(2*np.pi*Yu)*(np.cos(2*np.pi*Xu) - 2*(np.sin(np.pi*Xu)**2)) -\
             np.pi*np.sin(tn)*np.sin(np.pi*Yu)*np.sin(np.pi*Xu)

        Fy = -np.pi*np.cos(tn)*np.sin(2*np.pi*Xv)*(np.sin(np.pi*Yv)**2) -\
             2*(np.pi**3)*np.sin(tn)*np.sin(2*np.pi*Xv)*(2*(np.sin(np.pi*Yv)**2) - np.cos(2*np.pi*Yv)) +\
             np.pi*np.sin(tn)*np.cos(np.pi*Xv)*np.cos(np.pi*Yv)
        return [Fx[:,1:n], Fy[1:m,:]]

    def periodic_forcing_2(self): 
        # defines the first type of periodic forcing problem
        n = self.n
        m = self.m
        dx = self.dx
        dy = self.dy
        dt = self.dt
        t = self.t
        Re = self.Re
        Xu, Yu = np.meshgrid(self.xu, self.yu)
        Xv, Yv = np.meshgrid(self.xv, self.yv)      

        tn = dt*t + self.tdomain[0]
        # these forcing terms include boundary points
        Fx = np.cos(Xu + tn)*np.sin(Yu + tn) + np.cos(Yu + tn)*np.sin(Xu + tn) +\
             2*np.sin(Xu + tn)*np.sin(Yu + tn)+ np.cos(Xu - Yu + tn)
        Fy = -np.sin(Xv + tn)*np.cos(Yv + tn) - np.sin(Yv + tn)*np.cos(Xv + tn) +\
             2*np.cos(Xv + tn)*np.cos(Yv + tn)- np.cos(Xv - Yv + tn)

        return [Fx[:,1:n], Fy[1:m,:]]

    def select_forcing_term(self):
        # this functions selects the appropriate forcing term functions according to the name of the flow problem
        if self.test_problem_name == 'periodic_forcing_1':
            forcing_term = self.periodic_forcing_1()

        elif self.test_problem_name == 'periodic_forcing_2':
            forcing_term = self.periodic_forcing_2()

        else:
            forcing_term = 0
        return forcing_term
