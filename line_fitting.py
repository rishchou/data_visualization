
# @File - line_fitting.py
# @Author - Rishabh Choudhary
# @Description - Line fitting of the dataset using 1) Ordinary least squares 2) Total least squares
# 3) Least Sqaure + Regularization


import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy import linalg as la

# Function to plot fitted line
def abline(slope, intercept, fitType):
	"""Plot a line from slope and intercept"""
	axis = plt.gca()
	x = np.array(axis.get_xlim())
	y = intercept + (slope*x)
	plt.plot(x, y, '--', label = fitType)


# Ordinary Least square line fitting

def ordinary_least_square(X,Y):

	t = np.ones([len(X),1]) 
	X = np.concatenate((X.reshape(-1,1),t),axis=1)
	
	# B = (X'X)^-1(X'Y) for OLS.
	B = la.inv(X.transpose().dot(X)).dot(X.transpose().dot(Y))
	B = B.reshape(-1,1)
	
    # Slope of the fitted line
	m = B[0,:]
	print("Slope of LS fitted line:",m)
	#Intercept of the fitted line
	b = B[1,:]
	print("Intercept OF LS fitted line:",b)	
	#Plot the fitted line
	abline(m,b,"LS")
	plt.legend()

# Function to find U matrix elements for TLS.
def uTelement(x,y):
	xbar, ybar = x.mean(), y.mean()
	return np.sum((x - xbar)*(y - ybar))
	
# Function to find U matrix for TLS
def uTMatrix(x,y):
	uTmatrix = np.array([[uTelement(x,x), uTelement(x,y)],
						[uTelement(x,y), uTelement(y,y)]]) 
	return uTmatrix


#Function to compute Total least squares and fit TLS line
def total_least_square(X,Y):

	X = da[:,0]
	Y = da[:,1]

	uMatrix = uTMatrix(X,Y)
	
	# To find the eigen vector corresponding to the smallest eigenvalue
	lmda,v = la.eig(uMatrix)
	
	[a,b] = v[:,1]

	d = a*X.mean() + b*Y.mean()
	
	m = -(a/b)
	print("Slope of TLS fitted line:",m)
	b = d/b
	print("Intercept OF TLS fitted line:",b)

	abline(m,b,"TLS")
	plt.legend()


# LS + Regularization
def regularization(X,Y):
	p_factor = 180 
	t = np.ones([len(X),1]) 
	X = np.concatenate((X.reshape(-1,1),t),axis=1)

	#B = (X'X + p_factor(I))^-1 *(X'Y)
	B = la.inv(X.transpose().dot(X) + p_factor*np.identity(2)).dot(X.transpose().dot(Y))
	B = B.reshape(-1,1)
	m = B[0,:]
	print("Slope of LS+regularization fitted line:",m)

	b = B[1,:]
	print("Intercept OF LS+regularization fitted line:",b)
	abline(m,b,"LS+regularization")
	plt.legend()

k = int(input("enter the number(1-3) for the data file you want to do line fitting for: "))
if k==1:
	s = "data1_new.pkl"
elif k==2:
	s = "data2_new.pkl"
elif k==3:
	s = "data3_new.pkl"
else:
	print("Wrong input, choose between 1-3")
	exit()

#Load data file chosen
with open(s,'rb') as f:
	data = pickle.load(f)
da = np.asarray(data)

#Extract x and y data points
X = da[:,0]
Y = da[:,1]

#Plot the output figure
plt.figure("Output Window")

plt.plot(X,Y,'ro')
plt.axis('equal')

#Run the 3 methods for line fitting
ordinary_least_square(X,Y)
total_least_square(X,Y)
regularization(X,Y)
plt.show()

