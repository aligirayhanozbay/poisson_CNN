# -*- coding: utf-8 -*-
from __future__ import division
import sys
from mpl_toolkits.mplot3d import *
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
import structure
import solvers

def get_inputs():
	test_problem_dict = {1:'Taylor', 2:'periodic_forcing_1', 
				3:'periodic_forcing_2', 4:'driven_cavity'}
	test_problem_index = '1'#input('Choose the test problem from below [1:Taylor (default), 2:periodic_forcing_1, 3:periodic_forcing_2, 4:driven_cavity]: ')
	while test_problem_index == '':
		# take default
		test_problem_index = 1
	try:				
		test_problem_index = int(test_problem_index)
	except 	ValueError:
		test_problem_index = int(input('index of the test problem must be integers, try again: '))
	test_problem_name = test_problem_dict[test_problem_index]

	method_dict = {1:'Gauge', 2:'Alg1', 3:'Alg2', 4:'Alg3'}
	method_index = '1'#input('Choose the numerical algorithm from below [1:Gauge (default), 2:Alg1, 3:Alg2, 4:Alg3]: ')
	while method_index == '':
		# take default
		method_index = 1
	try:
		method_index = int(method_index)
	except ValueError:
		method_index = int(input('index of the method must be integers, try again: '))
	method = method_dict[method_index]

	space_input = str(0) + ',' + str(np.pi)#input('Enter the end points of the spatial domain (e.g. 0,1): ')
	while space_input == '':
		# take default (different for each problem)
		if test_problem_name == 'Taylor':
			space_input = [-np.pi/4.0, np.pi/4.0]
		elif test_problem_name == 'periodic_forcing_1':
			space_input = [-1,1]
		else:
			space_input = [0,1]
	try:
		xl, xr = space_input.split(',')
		xl = float(xl)
		xr = float(xr)
	except:
		if isinstance(space_input, list) == True:
			xl = space_input[0]
			xr = space_input[1]
		else:
			xl, xr = input('End points must be float or integers, separate them by a comma (,) now try again: ').split(',')
			xl = float(xl)
			xr = float(xr)
	
	time_input = '0,1'#input('Enter the start and end time (default: 0,1): ')
	while time_input == '':
		# take default
		time_input = '0,1'
	try:
		t0, tf = time_input.split(',')
		t0 = float(t0)
		tf = float(tf)
	except ValueError:
		time_input = input('Time must be float or integers, separate start and end time by a comma (,) now try again: ').split(',')
		t0 = float(time_input[0])
		tf = float(time_input[1])
	
	if test_problem_name == 'driven_cavity':
		pass
	else:
		Error_analysis_option = 'N'#input('Error analysis ? (Y/N) (default no) ')
		if 'Y' in Error_analysis_option or 'y' in Error_analysis_option:
			Error_analysis_option = True
			Niter = input('How many times you want to run the solver? (Default is 5 and this corresponds to grid size of 15, 30, 60, 120 and 240): ')
			while Niter == '':
				# take default value
				Niter = 5
			try:
				Niter = int(Niter)
			except ValueError:
				Niter = int(input('Number of runs must be an integer, try again: '))

			CFL, Re = opt_param()
			return xl, xr, t0, tf, Error_analysis_option, method, test_problem_name, CFL, Re, Niter
		elif Error_analysis_option == '':
			# take default
			Error_analysis_option = False
		else:
			Error_analysis_option = False

	gridsize = '100'#input('Enter the size of the spatial grid (default 30): ')
	while gridsize == '':
		# take default value
		gridsize = 30
	try:
		gridsize = int(gridsize)
	except ValueError:
		gridsize = int(input('Grid size must be integers, try again: '))

	plot_option = 'N'#input('plot results optional (Y/N): (default no)')
	if 'Y' in plot_option:
		plot_option = True
	elif 'y' in plot_option:
		plot_option = True
	elif plot_option == '':
		# take default
		plot_option = False
	else:
		plot_option = False
	
	CFL, Re = 0.1,1.0#opt_param()

	return xl, xr, t0, tf, gridsize, method, test_problem_name, plot_option, CFL, Re

def opt_param():
	optional_parameters = input('optional parameters e.g CFL number and Reynolds number? (Y/N) (default no)')
	if 'Y' in optional_parameters or 'y' in optional_parameters:
		CFL = input('CFL number (default 0.2): ')
		while CFL == '':
			# take default value
			CFL = 0.2
		try:
			CFL = float(CFL)
		except ValueError:
			CFL = float(input('CFL number must be a floating point number, try again: '))
	
		Re = input('Reynolds number (default 1): ')
		while Re == '':
			# take default value
			Re = 1.0
		try:
			Re = float(Re)
		except ValueError:
			Re = float(input('Reynolds number must be a floating point number or integer, try again: '))

	elif optional_parameters == '':
		# take default
		CFL = 0.1
		Re = 1.0
	else:
		# take default
		CFL = 0.1
		Re = 1.0
	return CFL, Re

def error_analysis(xl, xr, t0, tf, method, test_problem_name, CFL=0.1, Re=1.0, Niter=5):
	# Niter: number of iterations. e.g Niter = [15, 30, 60, 120, 240]
	U_convg = {}
	P_convg = {}
	log_dt = []
	start_grid = 15
	gridsizel = start_grid*(2**np.linspace(0, Niter-1, Niter))
	gridsizel = gridsizel.astype(np.int)

	for i in gridsizel:
		gridsize = i
		Velocity_error, Pressure_error, avg_gradp_error, dt = run_Navier_Stokes_solver(xl, xr, t0, tf, gridsize, method, test_problem_name, plot_option=False, CFL=CFL)
		UL1 = np.log(Velocity_error[0]['L1'])
		UL2 = np.log(Velocity_error[0]['L2'])
		ULinf = np.log(Velocity_error[0]['Linf'])
		U_convg.update({i:[UL1, UL2, ULinf]})

		PL1 = np.log(Pressure_error['L1'])
		PL2 = np.log(Pressure_error['L2'])
		PLinf = np.log(Pressure_error['Linf'])
		P_convg.update({i:[PL1, PL2, PLinf]})
		#break	
		log_dt.append(np.log(dt))
	
	# gsi: index for different gridsize: 15, 30, 60...
	log_UL1_error = np.array([U_convg[gsi][0] for gsi in gridsizel])
	U_slope_L1, U_intercept_L1, r_value, p_value, std_err = stats.linregress(np.array(log_dt), log_UL1_error)
	print(U_slope_L1, 'U velocity L1 convergence rate is %s' % U_slope_L1)
	log_UL2_error = np.array([U_convg[gsi][1] for gsi in gridsizel])
	U_slope_L2, U_intercept_L2, r_value, p_value, std_err = stats.linregress(np.array(log_dt), log_UL2_error)
	print(U_slope_L2, 'U velocity L2 convergence rate is %s' % U_slope_L2)
	log_ULinf_error = np.array([U_convg[gsi][2] for gsi in gridsizel])
	U_slope_Linf, U_intercept_Linf, r_value, p_value, std_err = stats.linregress(np.array(log_dt), log_ULinf_error)
	print(U_slope_Linf, 'U velocity Linf convergence rate is %s' % U_slope_Linf)

	# only take grids bigger than 15, because the errors approaching the limit asymptotically and a too small grid is not very useful (especially for pressure)
	log_PL1_error = np.array([P_convg[gsi][0] for gsi in gridsizel])
	P_slope_L1, P_intercept_L1, r_value, p_value, std_err = stats.linregress(np.array(log_dt)[1:], log_PL1_error[1:])
	#P_slope_L1, P_intercept_L1, r_value, p_value, std_err = stats.linregress(np.array(log_dt), log_PL1_error)
	print(P_slope_L1, 'P L1 convergence rate is %s' % P_slope_L1)
	log_PL2_error = np.array([P_convg[gsi][1] for gsi in gridsizel])
	P_slope_L2, P_intercept_L2, r_value, p_value, std_err = stats.linregress(np.array(log_dt)[1:], log_PL2_error[1:])
	#P_slope_L2, P_intercept_L2, r_value, p_value, std_err = stats.linregress(np.array(log_dt), log_PL2_error)
	print(P_slope_L2, 'P L2 convergence rate is %s' % P_slope_L2)
	log_PLinf_error =  np.array([P_convg[gsi][2] for gsi in gridsizel])
	P_slope_Linf, P_intercept_Linf, r_value, p_value, std_err = stats.linregress(np.array(log_dt)[1:], log_PLinf_error[1:])
	#P_slope_Linf, P_intercept_Linf, r_value, p_value, std_err = stats.linregress(np.array(log_dt), log_PLinf_error)
	print(P_slope_Linf, 'P Linf convergence rate is %s' % P_slope_Linf)

	log_dt = np.array(log_dt)

	# convergence plots
	# for velocity (U component, V is the same)
	plt.figure(1)
	plt.plot(log_dt, log_dt*U_slope_L1+U_intercept_L1, label='L1 velocity error, convergence rate = %s' % U_slope_L1, color='r')
	plt.scatter(log_dt, log_UL1_error, color='r')
	plt.plot(log_dt, log_dt*U_slope_L2+U_intercept_L2, label='L2 velocity error, convergence rate = %s' % U_slope_L2, color='b')
	plt.scatter(log_dt, log_UL2_error, color='b')
	plt.plot(log_dt, log_dt*U_slope_Linf+U_intercept_Linf, label='Linf velocity error, convergence rate = %s' % U_slope_Linf, color='g')
	plt.scatter(log_dt, log_ULinf_error, color='g')
	plt.plot(log_dt, log_dt*2 + min([U_intercept_L1, U_intercept_L2, U_intercept_Linf])-1, '--', label='2nd order reference line', color='k', linewidth=1.5)
	plt.xlabel('log(dt)')
	plt.ylabel('log(error)')
	plt.title('log log plot of the temporal error convergence rates for velocity, method: ' + method+', flow problem: '+ test_problem_name+', CFL = %s' % CFL)
	plt.legend(loc=4)

	# for pressure
	plt.figure(2)
	plt.plot(log_dt, log_dt*P_slope_L1+P_intercept_L1+3, label='L1 pressure error, convergence rate = %s' % P_slope_L1, color='r')
	plt.scatter(log_dt, log_PL1_error+3, color='r')
	plt.plot(log_dt, log_dt*P_slope_L2+P_intercept_L2, label='L2 pressure error, convergence rate = %s' % P_slope_L2, color='b')
	plt.scatter(log_dt, log_PL2_error, color='b')
	plt.plot(log_dt, log_dt*P_slope_Linf+P_intercept_Linf-3, label='Linf pressure error, convergence rate = %s' % P_slope_Linf, color='g')
	plt.scatter(log_dt, log_PLinf_error-3, color='g')
	plt.plot(log_dt, log_dt*2 + min([P_intercept_L1+3, P_intercept_L2, P_intercept_Linf-1])-3, '--', label='2nd order reference line', color='k', linewidth=1.5)
	plt.xlabel('log(dt)')
	plt.ylabel('log(error)')
	plt.title('log log plot of the temporal error convergence rates for pressure, method: '+ method+', flow problem: '+ test_problem_name+', CFL = %s' % CFL)
	plt.legend(loc=4)

	plt.show()

def run_Navier_Stokes_solver(xl, xr, t0, tf, gridsize, method, test_problem_name, plot_option, CFL=0.1, Re=1.0):
	grid_size_domain = [gridsize, gridsize]
	m, n = grid_size_domain
	spatial_domain = [[xl,xr],[xl,xr]]
	time_domain = [t0,tf]
	print('start')
	mesh = structure.mesh(grid_size_domain,spatial_domain,time_domain,CFL,Re)
	print(mesh.dx, "dx")
	print(mesh.dt, "dt")
	print(spatial_domain, 'spatial domain')
	print(time_domain, 'time domain')
	Xubnd = mesh.ubndmg("x")
	Yubnd = mesh.ubndmg("y")
	Xvbnd = mesh.vbndmg("x")
	Yvbnd = mesh.vbndmg("y")
	XPint = mesh.pintmg("x")
	YPint = mesh.pintmg("y")
	tend = mesh.tdomain[1]

	if method == 'Gauge':
		ic_uv_init = structure.InitialCondition(mesh).select_initial_conditions(test_problem_name)[0]
		# use Gauge method
		Gauge = solvers.Gauge_method(Re, mesh)
		# initial set up
		init_setup = Gauge.setup(ic_uv_init, test_problem_name)
		# iterative solve process
		uvf_cmp, pf, gradp = Gauge.iterative_solver(test_problem_name, mesh.Tn, init_setup)
	
	elif method == 'Alg1':
		ic_init = structure.InitialCondition(mesh).select_initial_conditions(test_problem_name)
		# use Alg 1 method
		Alg1 = solvers.Alg1_method(Re, mesh)
		# initial set up
		init_setup = Alg1.setup(ic_init, test_problem_name)
		# iterative solve process
		uvf_cmp, pf, gradp = Alg1.iterative_solver(test_problem_name, mesh.Tn, init_setup)
	
	elif method == 'Alg2':
		# use Alg 2 
		ic_uv_init = structure.InitialCondition(mesh).select_initial_conditions(test_problem_name)
		Alg2 = solvers.Alg2_method(Re, mesh)
		# initial set up
		init_setup = Alg2.setup(ic_uv_init, test_problem_name)
		# iterative solve process
		uvf_cmp, pf, gradp = Alg2.iterative_solver(test_problem_name, mesh.Tn, init_setup)
	
	elif method == 'Alg3':
		# use Alg 3 (pressure free projection method)
		ic_init = structure.InitialCondition(mesh).select_initial_conditions(test_problem_name)[0]
		# use Alg1 method
		Alg3 = solvers.Alg3_method(Re, mesh)
		# initial set up
		init_setup = Alg3.setup(ic_init, test_problem_name)
		# iterative solve process
		uvf_cmp, pf, gradp = Alg3.iterative_solver(test_problem_name, mesh.Tn, init_setup)
	
	# comparison and error analysis
	if test_problem_name == 'driven_cavity':
		# no analytical solutions available
		pass
	else:
		uv_exact_bnd, p_exact, gradp_exact = structure.Exact_solutions(mesh, Re, mesh.Tn).Exact_solutions(test_problem_name)
		div_uvf = uvf_cmp.divergence()
		print(mesh.integrate(p_exact), 'integral of exact pressure')
		Error = solvers.Error(uvf_cmp, uv_exact_bnd, pf, p_exact, gradp, gradp_exact, div_uvf, mesh)
		Velocity_error = Error.velocity_error()
		Pressure_error = Error.pressure_error()
		gradpu_error, gradpv_error, avg_gradp_error = Error.pressure_gradient_error()
	print(test_problem_name)
	print(method)
	print('CFL = %s' % CFL)
	
	if plot_option == False:
		return Velocity_error, Pressure_error, avg_gradp_error, mesh.dt
	else:
		uf_bnd = uvf_cmp.get_bnd_uv()[0]
		vf_bnd = uvf_cmp.get_bnd_uv()[1]
		
		fig1 = plt.figure(figsize=(6,6), dpi=100)
		ax1 = fig1.gca(projection='3d')
		ax1.plot_surface(Xubnd,Yubnd,uf_bnd,rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
		ax1.set_xlabel('x')
		ax1.set_ylabel('y')
		ax1.set_title('Plot of Numerical U Velocity at Time = '+str(tend))
		#plt.savefig('Gauge_Taylor_U_exact_grid_'+str(n)+'.jpg', bbox_inches='tight')
		#plt.savefig('Gauge_unf1_U_exact_grid_'+str(n)+'.pdf', bbox_inches='tight')
		
		fig2 = plt.figure(figsize=(6,6), dpi=100)
		ax2 = fig2.gca(projection='3d')
		ax2.plot_surface(Xvbnd,Yvbnd,vf_bnd,rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
		ax2.set_xlabel('x')
		ax2.set_ylabel('y')
		ax2.set_title('Plot of Numerical V Velocity at Time = '+str(tend))
		#plt.savefig('Gauge_Taylor_V_exact_grid_'+str(n)+'.jpg', bbox_inches='tight')
		#plt.savefig('Gauge_Taylor_V_exact_grid_'+str(n)+'.pdf', bbox_inches='tight')
		
		fig3 = plt.figure(figsize=(6,6), dpi=100)
		ax3 = fig3.gca(projection='3d')
		ax3.plot_surface(XPint, YPint, pf.get_value() ,rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
		ax3.set_xlabel('x')
		ax3.set_ylabel('y')
		ax3.set_title('Plot of Numerical Pressure at Time = '+str(tend))
		#plt.savefig('Gauge_Taylor_pf_grid_'+str(n)+'.jpg', bbox_inches='tight')
		#plt.savefig('Gauge_Taylor_pf_grid_'+str(n)+'.pdf', bbox_inches='tight')
		
		if test_problem_name == 'driven_cavity':
			# no analytical solutions availabe
			pass
		else:
			fig4 = plt.figure(figsize=(6,6), dpi=100)
			ax4 = fig4.gca(projection='3d')
			ax4.plot_surface(Xubnd,Yubnd,uv_exact_bnd.get_uv()[0],rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
			ax4.set_xlabel('x')
			ax4.set_ylabel('y')
			ax4.set_title('Plot of Analytical U Velocity at Time = '+str(tend))
			#plt.savefig('Gauge_Taylor_U_exact_grid_'+str(n)+'.jpg', bbox_inches='tight')
			#plt.savefig('Gauge_Taylor_U_exact_grid_'+str(n)+'.pdf', bbox_inches='tight')
			
			fig5 = plt.figure(figsize=(6,6), dpi=100)
			ax5 = fig5.gca(projection='3d')
			ax5.plot_surface(Xvbnd,Yvbnd,uv_exact_bnd.get_uv()[1],rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
			ax5.set_xlabel('x')
			ax5.set_ylabel('y')
			ax5.set_title('Plot of Analytical V Velocity at Time = '+str(tend))
			#plt.savefig('Gauge_Taylor_V_exact_grid_'+str(n)+'.jpg', bbox_inches='tight')
			#plt.savefig('Gauge_Taylor_V_exact_grid_'+str(n)+'.pdf', bbox_inches='tight')
			
			fig6 = plt.figure(figsize=(6,6), dpi=100)
			ax6 = fig6.gca(projection='3d')
			ax6.plot_surface(XPint,YPint,p_exact.get_value(),rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
			ax6.set_xlabel('x')
			ax6.set_ylabel('y')
			ax6.set_title('Plot of Analytical Pressure at Time = '+str(tend))
			#plt.savefig('Gauge_Taylor_P_exact_grid_'+str(n)+'.jpg', bbox_inches='tight')
			#plt.savefig('Gauge_Tyalor_P_exact_grid_'+str(n)+'.pdf', bbox_inches='tight')
			##
			fig7 = plt.figure(figsize=(6,6), dpi=100)
			ax7 = fig7.gca(projection='3d')
			ax7.plot_surface(XPint,YPint,pf.get_value()-p_exact.get_value(),rstride=1, cstride=1, cmap=cm.jet, linewidth=0.2)
			ax7.set_xlabel('x')
			ax7.set_ylabel('y')
			ax7.set_title('Plot of Pressure error at Time = '+str(tend))
			#plt.savefig('Gauge_Taylor_P_exact_grid_'+str(n)+'.jpg', bbox_inches='tight')
			#plt.savefig('Gauge_Tyalor_P_exact_grid_'+str(n)+'.pdf', bbox_inches='tight')
		
		plt.show()
		return Velocity_error, Pressure_error, avg_gradp_error, mesh.dt

if __name__ == "__main__":
	print('WARNING: At the moment, this simulation code uses hard coded file paths in the solvers.py file and the usage of the model is commented out (line 250 thereabouts). Change these to use this script properly!')
	inputs = get_inputs()
	if type(inputs[4]) == bool:
		xl, xr, t0, tf, gridsize, method, test_problem_name, CFL, Re, Niter = inputs
		error_analysis(xl, xr, t0, tf, method, test_problem_name, CFL, Re, Niter)
	else:
		xl, xr, t0, tf, gridsize, method, test_problem_name, plot_option, CFL, Re = inputs
		Velocity_error, Pressure_error, avg_gradp_error, dt = run_Navier_Stokes_solver(xl, xr, t0, tf, gridsize, method, test_problem_name, plot_option, CFL, Re)
		print("U velocity error is %s " % Velocity_error[0])
		print("V velocity error is %s " % Velocity_error[1])
		print("Pressure error is %s " % Pressure_error)
		print("average gradient Pressure error is %s " % avg_gradp_error)

