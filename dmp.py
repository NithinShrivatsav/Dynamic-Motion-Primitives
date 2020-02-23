'''
Author: Nithin Shrivatsav Srikanth
Description: This python script runs dynamic motion primitives
'''
import pandas 
import numpy as np
import matplotlib.pyplot as plt

def psi_value(x_t, t, alpha_x, h, c):
	return np.exp(-h*((x_t-c)**2))

def return_sum_gaussians(x_t, t, h, c, m, alpha_x):
	bases_func_sum = 0
	for i in range(m):
		bases_func_sum += psi_value(x_t, t, alpha_x, h, c[i])
	return bases_func_sum

def return_weighted_sum_gaussians(x_t, w, t, h, c, m, alpha_x):
	bases_func_weighted_sum = 0
	for i in range(m):
		bases_func_weighted_sum += w[i]*psi_value(x_t, t, alpha_x, h, c[i])
	return bases_func_weighted_sum

def calculate_s(x0, yg, y0, trajectory_size):
	s = []
	dt = 0.01
	x_t = x0
	for j in range(trajectory_size):
		s_element = x_t*(yg-y0)
		x_t += -alpha_x*x_t*dt
		s.append(s_element)
	s = np.array(s)
	s = s.reshape(trajectory_size,1)
	return s

def generate_gaussian_kernel_matrix(m, x0, trajectory_size, alpha_x, h, c):
	psi = []
	dt = 0.01
	for k in range(m):
		psi_k = []
		x_t = x0
		for t in range(trajectory_size):
			psi_k_t = psi_value(x_t, t, alpha_x, h, c[k])
			x_t += -alpha_x*x_t*dt
			psi_k.append(psi_k_t)
		psi.append(np.diag(psi_k))
	return np.array(psi)

if __name__=="__main__":
	## Read the trajectory dataset
	y = pandas.read_csv("slice-y-sample.csv")
	dy = pandas.read_csv("slice-dy-sample.csv")
	trajectory_size = y.shape[0]

	## Extract the trajectory value for the 7th joint
	y_7 = np.array(y.iloc[:, [6]])
	dy_7 = np.array(dy.iloc[:, [6]])
	y_7_g = y_7[trajectory_size-1]
	dy_7_g = dy_7[trajectory_size-1]

	## Parameters of the problem
	m = 8 ## no of gaussian kernels
	alpha_y = 8 ## first coefficient of point-attractor system 
	beta_y = 10 ## second coefficient of point-attractor system
	alpha_x = 1 ## coefficient of canonical system
	dt = 1/100

	## Parameters of the gaussian kernels
	h = 1
	c = [1/8, 2/8, 3/8, 4/8, 5/8, 6/8, 7/8, 1] 

	## initial state of canonical dynamical system
	x_0 = 1.0
	dt = 0.01

	## Compute the acceleration of the system
	ddy_7 = []
	ddy_7.append(0)
	for i in range(1, trajectory_size):
		ddy_temp = ((dy_7[i] + dy_7[i-1])/2.0)
		ddy_7.append(ddy_temp)

	ddy_7 = np.array(ddy_7)
	ddy_7 = ddy_7.reshape(trajectory_size,1)


	## Generate the s vector
	s = calculate_s(x_0, y_7_g, y_7[0], trajectory_size)

	## Generate the gaussian kernel matrix
	psi_matrix = generate_gaussian_kernel_matrix(m, x_0, trajectory_size, alpha_x, h, c)

	## Generate desired trajectory vector
	forcing_function = []
	for l in range(trajectory_size):
		forcing_function_element = ddy_7[l] - alpha_y*(beta_y*(y_7_g-y_7[l]) - dy_7[l])
		forcing_function.append(forcing_function_element)

	forcing_function = np.array(forcing_function)

	## Calculate the weights
	weights = []
	for w in range(m):
		weights_temp = (np.dot(np.dot(s.T,psi_matrix[w]),forcing_function))/(np.dot(np.dot(s.T,psi_matrix[w]),s))
		weights_temp = weights_temp.ravel()
		weights.append(weights_temp[0][0])
	
	weights = np.array(weights)
	weights = weights.reshape(weights.shape[0],1)

	## Generate the trajectory
	y_generated = []
	dy_generated = []
	ddy_generated = []

	y_generated.append(y_7[0])
	dy_generated.append(dy_7[0])
	ddy_generated.append(ddy_7[0])

	x_t = x_0
	for t in range(1,trajectory_size):
		ddy_element = alpha_y*(beta_y*(y_7_g-y_generated[t-1]) - dy_generated[t-1]) + ((return_weighted_sum_gaussians(x_t, weights, t, h, c, m, alpha_x)/return_sum_gaussians(x_t, t, h, c, m, alpha_x))*x_t*(y_7_g-y_7[0]))
		dy_element = dy_generated[t-1] + ddy_generated[t-1]*dt
		y_element = y_generated[t-1] + dy_generated[t-1]*dt
		y_generated.append(y_element)
		dy_generated.append(dy_element)
		ddy_generated.append(ddy_element)
		x_t += -alpha_x*x_t*dt

	plt.plot(np.ones(len(y_7))*y_7_g, 'r--', lw=2, c='r')
	plt.plot(y_7, lw=2, c='g')
	plt.plot(y_generated, lw=2, c='b')
	plt.title('DMP')
	plt.xlabel('time (1/100 of s)')
	plt.ylabel('trajectory')
	plt.legend(['goal', 'demonstration trajectory', 'executed trajectory'], loc='upper right')
	plt.show()
