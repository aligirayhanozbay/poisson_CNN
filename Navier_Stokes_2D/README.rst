
Navier Stokes 2D
================


Navier_Stokes_2D is an open-source software used to solve 2D Navier Stokes equations on a uniform square domain. This software is the extension of Hongji's honours project. This software implements the popular Projection method (originally developed independently by A. J. Chorin and R. Temam) to solver the 2D Navier Stokes equations using finite difference discretisation.

Navier_Stokes_2D is a Python package. The recommended version is Python 2.7

Our code covers various popular second order projection methods, such as Incremental pressure project methods, Pressure free method and the Gauge method. 

Developed at Mathematical Sciences Institute at the Australian National University.

Copyright 2015 Stephen Roberts, Hongji Zhang

Installation
------------

You can simply download the 3 files: solvers.py, structure.py and run_solvers.py. However you will need to install the dependencies first. They are: matplotlib (version >= 1.3.1), numpy (version >= 1.8.2) and scipy (version >= 0.14.0).

Introduction
------------

You can run a couple of Navier Stokes problem simulations by running the script run_solvers.py. Then a user interface will appear in your command line. It asks you basic input information, such as the test fluid flow problem, type of projection method and spatial and temporal domain ... Just follow the prompts. If you hit enter without entering any value, Navier_Stokes_2D will take the default values.

You can either run accuracy tests for projection methods or you can just run simulations of particular fluid flow problems with an arbitrary domain and precision (controled by spatial grid size). If you run accuracy tests, then the solver will run for several times with grid size doubled each time, and you will be presented with the convergence test results for both velocity and pressure. If you run direct simulations, you will be presented with the 3D surface plots of velocity and pressure as well as the pressure error plot (if applicable).

Projection methods
------------------

Projection methods first proposed by A. J. Chorin and R. Temam independently inthe 1960s still remain arguably the most popular numerical methods for solving incompressible Navier Stokes equations. The difficulty with solving incompressible Navier Stokes equations lies in the part where the velocity and pressure are coupled through the incompressibility constraint. Projection method overcomes this problem by solving them indepently and projecting the final velocity into the space of divergence vector fields. Thus projection methods turn the problem into an iterative solve process. This is why they are often coined as the incremental projection methods. 

The popular "second order" projection methods are:

 1. A well known incremental projection method is proposed by Bell, Colella, and Glaz. This is second order in space and time for velocity but only first order in pressure in L infinity norm. This is implemented as "Alg 1" method in this software.

 2. A modification of this method proposed by David Brown improves the pressure convergence to about 1.5 - 2nd order. This is implemented as "Alg 2" method in this software.

 3. The pressure free projection method proposed by Kim and Moin is also very popular. The pressure variable is not involved in the iterative solve process. It usually demonstrates second order convergence for both velocity and pressure in most uniform grids. This is implemented as "Alg 3" method in this software.

 4. Gauge method is a recent popular method where the pressure and velocity variables are demonstrated to be both second order accurate. The method works by replacing velocity and pressure by Gauge variable and an auxiliary scalar field. This is implemented as "Gauge" method in this software.

For more details about how these methods work and their theoretical and numerical convergence analysis, please refer to my thesis, or you can find a lot of papers on this topic in Google.

Enjoy!


