from __future__ import print_function
import argparse
import os
import json
from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import tensorflow as tf
import poisson_CNN

parameters['reorder_dofs_serial']=False

# class GridInterpolantExpression(UserExpression):
#     def __init__(self, grid_values, domain_extent,**spline_options):
#         x = np.linspace(*domain_extent[0],grid_values.shape[-2])
#         y = np.linspace(*domain_extent[1],grid_values.shape[-1])
#         z = np.reshape(grid_values, [grid_values.shape[-2], grid_values.shape[-1]])
#         bbox = [x[0],x[-1],y[0],y[-1]]
#         self._spline = RectBivariateSpline(x,y,z,bbox=bbox,**spline_options)

#         super().__init__()

#     def value_shape(self):
#         return tuple()

#     def eval(self, value, x):
#         value[0] = self._spline(*x)

def get_mesh_coordinate_indices(mesh, npts):
    coords = mesh.coordinates().transpose((1,0))
    indices = []
    for dim_npts, dim_coords in zip(npts, coords):
        indices.append(
            np.round(dim_coords/max(dim_coords) * (dim_npts-1)).astype(np.int32)
        )
    return np.stack(indices,-1)

parser = argparse.ArgumentParser(description = 'Run a Taylor-Green Vortex simulation using the projection method, with initial guesses to the pressure step linear solver provided by the Homogeneous Poisson NN model')
parser.add_argument('--model_config', type=str, help='Experiment JSON file containing the config of the HPNN model. If not provided, the program will be run without the NN.', default = None)
parser.add_argument('--model_checkpoint', type=str, help='Path to the Tensorflow checkpoint file containing model weights', default = None)
parser.add_argument('--output_folder', type=str, help='Folder for the output data')
args = parser.parse_args()
        
comm = MPI.comm_world

T = 1.0            # final time
num_steps = 5000   # number of time steps
dt = T / num_steps # time step size
mu = 6.25e-4         # dynamic viscosity
rho = 1            # density


#mesh_file = "square_valid.h5"
mesh_file = os.path.dirname(__file__) + "/square_valid.h5"

mesh = Mesh()
mesh_npts = [100,100]
try:
    with HDF5File(comm, mesh_file, "r") as h5file:
        h5file.read(mesh, "mesh", False)
        facet = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        h5file.read(facet, "facet")
except FileNotFoundError as fnf_error:
    print(fnf_error)

mesh_indices = get_mesh_coordinate_indices(mesh, mesh_npts)
grid_spacing = sorted(list(set(mesh.coordinates()[:,0][1:] - mesh.coordinates()[:,0][:-1])))[-1]

# coords = mesh.coordinates()
# plt.plot(coords[:,0],coords[:,1])
# plt.savefig('/storage/fenics_cfd_test/coords.png', dpi = 400)
# plt.figure()
# indices = get_mesh_coordinate_indices(mesh, [100,100])
# plt.plot(indices[:,0],indices[:,1])
# plt.savefig('/storage/fenics_cfd_test/indices.png', dpi = 400)

ExactU = Expression(('cos(x[0]) * sin(x[1])', '-sin(x[0]) * cos(x[1])'), degree = 1)


# Define function spaces
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)

topBC = DirichletBC(V, ExactU, facet, 1)
rightBC = DirichletBC(V, ExactU,facet, 2)
botBC = DirichletBC(V, ExactU, facet, 3)
leftBC = DirichletBC(V, ExactU, facet, 4)

bcu = [topBC, rightBC, botBC, leftBC]


# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V)
u_  = Function(V)
p_n = Function(Q)
p_  = Function(Q)
p_nn = Function(Q)
# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx \
   + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
   + inner(sigma(U, p_n), epsilon(v))*dx \
   + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
   - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

[bc.apply(A1) for bc in bcu]

initial_condition_u = Expression(('cos(x[0]) * sin(x[1])', '-sin(x[0]) * cos(x[1])'), degree = 1)
initial_condition_u = project(initial_condition_u, V)
assign(u_n, initial_condition_u)
assign(u_, initial_condition_u)

# Create XDMF files for visualization output
folder='/storage/fenics_cfd_test/'
xdmffile_u = XDMFFile(folder + '/velocity.xdmf')
xdmffile_p = XDMFFile(folder + '/pressure.xdmf')

# Create time series (for use in reaction_system.py)
timeseries_u = TimeSeries(folder + '/velocity_series')
timeseries_p = TimeSeries(folder + '/pressure_series')

# Save mesh to file (for use in reaction_system.py)
File(folder + '/cylinder.xml.gz') << mesh

#load model
model_config = poisson_CNN.convert_tf_object_names(json.load(open(args.model_config)))['model']
model = poisson_CNN.models.Homogeneous_Poisson_NN_Legacy(**model_config)
_ = model([tf.random.uniform((2,1,100,100)), tf.random.uniform((2,1))])
model.load_weights(args.model_checkpoint)

# Time-stepping
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    print('=====================')
    print('Step ' + str(n) + ' | Time ' + str(t))

    # Step 1: Tentative velocity step
    b1 = assemble(L1)
    [bc.apply(b1) for bc in bcu]
    solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

    # Step 2: Pressure correction step
    b2 = assemble(L2)
    if args.model_config is not None and n > 50:
        rhs = tf.zeros(mesh_npts)
        rhs = tf.tensor_scatter_nd_update(rhs, mesh_indices, np.array(b2))
        rhs_max_magnitude = tf.reduce_max(tf.abs(rhs))
        rhs = tf.expand_dims(tf.expand_dims(rhs,0),0)/rhs_max_magnitude
        dx = tf.constant([[grid_spacing]], tf.keras.backend.floatx())
        nn_pred = -model([rhs, dx])[0,0]*(np.pi**2)
        nn_pred = nn_pred/(tf.reduce_max(nn_pred) - tf.reduce_min(nn_pred)) - tf.reduce_min(nn_pred)
        #p_nn.vector().set_local(tf.gather_nd(nn_pred, mesh_indices))
        p_.vector().set_local(tf.gather_nd(nn_pred, mesh_indices))

        fig = plt.imshow(rhs[0,0], origin='lower') 
        plt.colorbar()
        plt.savefig(folder + '/rhs.png')
        plt.close()

        fig = plt.imshow(nn_pred, origin='lower') 
        plt.colorbar()
        plt.savefig(folder + '/pred.png')
        plt.close()

        # m_nn = project(div(grad(p_nn)),Q).vector() - b2
        # print('L2 norm p NN: ' + str(np.sum(np.array(m_nn)**2)))
    else:
        solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')
    
    if n > 60:
        import pdb
        pdb.set_trace()
    # m = project(div(grad(p_)),Q).vector() - b2
    # print('L2 norm p amg: ' + str(np.sum(np.array(m)**2)))
    
    # Step 3: Velocity correction step
    b3 = assemble(L3)
    solve(A3, u_.vector(), b3, 'cg', 'sor')

    # Plot solution
    myplot = plot(u_, title='Velocity at t=' + str(t))
    plt.colorbar(myplot)
    plt.savefig(folder + '/velocity.png', dpi = 400)
    plt.close()

    myplotp = plot(p_, title='Pressure at t=' + str(t))#, vmin = -0.5, vmax = 0.5)
    plt.colorbar(myplotp)
    plt.savefig(folder + '/pressure.png', dpi = 400)
    plt.close()

    # Save solution to file (XDMF/HDF5)
    xdmffile_u.write(u_, t)
    xdmffile_p.write(p_, t)

    # Save nodal values to file
    timeseries_u.store(u_.vector(), t)
    timeseries_p.store(p_.vector(), t)

    # Update previous solution
    u_n.assign(u_)
    p_n.assign(p_)

