#!/usr/bin/env python

# @File - outlier_rejection.py
# @Author - Rishabh Choudhary
# @Description - RANSAC and z-score implementation for outlier rejection of given datasets.

import numpy as np
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import sys
import math

#Z_Scores implementation
def z_scores(s):
	with open(s,'rb') as f:
		data = pickle.load(f)
	x,y = zip(*data)
	x = np.array(x)
	y = np.array(y)
	threshold = 2 # number of standrd devaition beyond which the point is outlier
	mean_y = np.mean(y) # mean of y
	stdev_y = np.std(y) # standard deviation of y
	num = 0
	z_outlier = np.empty((len(x),2))
	for i in range(len(x)):
		z_scores = (y[i]-mean_y)/(stdev_y)
		if abs(z_scores) > threshold: #compare the absolute value
			num = num + 1
			z_outlier = np.append(x[i],y[i]) # Store all the outlier in z_outlier array
	print(z_outlier.shape)
	print("the number of outliers it could find are " + str(num))
	#print(z_outlier)
	exit()

#Line model using points function
def find_line_model(points):

    m = (points[1,1] - points[0,1]) / (points[1,0] - points[0,0] + sys.float_info.epsilon) # slope
    c = points[1,1] - m * points[1,0]  # y-intercept of the line
 
    return m, c

#find the intersection point function
def find_intercept_point(m, c, x0, y0):
 
    # intersection point with the model
    x = (x0 + m*y0 - m*c)/(1 + m**2)
    y = (m*x0 + (m*2)*y0 - (m*2)*c)/(1 + m*2) + c
 
    return x, y

#Take the user input
a = int(input("enter 1 for RANSAC and enter 2 for `ore: "))

if a==1:
	k = int(input("enter the number(1-3) for the data file you want to do RANSAC for: "))
	if k==1:
		s = "data1_new.pkl"
		optimize_ratio = 0.7
		ran_ts = 5
	elif k==2:
		s = "data2_new.pkl"
		optimize_ratio = 0.5
		ran_ts = 7
	elif k==3:
		s = "data3_new.pkl"
		optimize_ratio = 0.4
		ran_ts = 20
	else:
		print("Wrong input, choose between 1-3")
		exit()
elif a==2:
	k = int(input("enter the number(1-3) for the data file you want to do Z-scores for: "))
	if k==1:
		s = "data1_new.pkl"
	elif k==2:
		s = "data2_new.pkl"
	elif k==3:
		s = "data3_new.pkl"
	else:
		print("Wrong input, choose between 1-3")
		exit()
	z_scores(s)

else:
	print("Wrong input, choose between 1-2")
	exit()

   
with open(s,'rb') as f:
    data = pickle.load(f)
x,y = zip(*data)


ransac_iterations = 20  # number of iterations
ransac_threshold = ran_ts    # threshold
ransac_ratio = optimize_ratio
n_samples = len(x)

#plot the ransac function
def ransac_plot(n, x, y, m, c, x_in, y_in):
	plt.figure("Ransac", figsize=(15., 15.))
 
    	# grid for the plot
	grid = [min(x) - 10, max(x) + 10, min(y) - 20, max(y) + 20]
	plt.axis(grid)
 
    	# plot input points
	plt.plot(x[:], y[:], marker='o', label='Input points', color='#00cc00', linestyle='None', alpha=0.4)
 
    	# draw the current model
	plt.plot(x, m*x + c, 'r', label='Line model', color='#ff0000', linewidth=3)
 
	plt.plot(x_in, y_in, marker='o', label='Inliers', linestyle='None', color='#ff0000', alpha=0.6)
	plt.legend()
	plt.show()
	

mat_data = np.zeros((len(x),2))
mat_data[:,0] = np.array(x)
mat_data[:,1] = np.array(y)
    
ratio = 0.
model_m = 0.
model_c = 0.

# perform RANSAC iterations
for it in range(ransac_iterations):
 
    # pick up two random points
    n = 2
    x = np.array(x)
    all_indices = np.arange(x.shape[0])
    np.random.shuffle(all_indices)
 
    indices_1 = all_indices[:n]
    indices_2 = all_indices[n:]
 
    maybe_points = mat_data[indices_1,:]
    test_points = mat_data[indices_2,:]
 
    # find a line model for these points
    m, c = find_line_model(maybe_points)
 
    x_list = []
    y_list = []
    num = 0
 
    # find orthogonal lines to the model for all testing points
    for ind in range(test_points.shape[0]):
 
        x0 = test_points[ind,0]
        y0 = test_points[ind,1]
 
        # find an intercept point of the model with a normal from point (x0,y0)
        x1, y1 = find_intercept_point(m, c, x0, y0)
 
        # distance from point to the model
        dist = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
 
        # check whether it's an inlier or not
        if dist < ransac_threshold:
            x_list.append(x0)
            y_list.append(y0)
            num += 1
 
    x_inliers = np.array(x_list)
    y_inliers = np.array(y_list)
    
    # in case a new model is better
    if num/float(n_samples) > ratio:
        ratio = num/float(n_samples)
        model_m = m
        model_c = c
 
    # we are done in case we have enough inliers
    if num > n_samples*ransac_ratio:
        print ('The model is found !')
        break
 
# plot the final model
ransac_plot(0, x,y, model_m, model_c, x_inliers,y_inliers)






