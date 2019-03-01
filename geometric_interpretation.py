#!/usr/bin/env python

# @File - geometric_interpretation.py
# @Author - Rishabh Choudhary
# @Description - Geometric interpretation and decomposition of data into eigen vectors (PCA)


import pickle
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math

def covariance(x,y):
    xbar, ybar = x.mean(), y.mean()
    return np.sum((x - xbar)*(y - ybar))/(len(x)-1)
    
#Covariance matrix
def covMat(x):
    covMat = np.array([[covariance(x[0],x[0]), covariance(x[0], x[1])], 
                        [covariance(x[1],x[0]), covariance(x[1],x[1])]])
    return covMat

def eigen_values(cov_mat):
    e_value[0] = (cov_mat[0,0]+cov_mat[1,1]+math.sqrt(cov_mat[0,0]**2+cov_mat[1,1]**2+4*cov_mat[0,1]*cov_mat[1,0]-2*cov_mat[0,0]*cov_mat[1,1]))/2
    e_value[1] = (cov_mat[0,0]+cov_mat[1,1]-math.sqrt(cov_mat[0,0]**2+cov_mat[1,1]**2+4*cov_mat[0,1]*cov_mat[1,0]-2*cov_mat[0,0]*cov_mat[1,1]))/2
    e_vector[0,0] = -cov_mat[0,1]/(cov_mat[0,0]-e_value[0])
    e_vector[0,1] = -cov_mat[0,1]/(cov_mat[0,0]-e_value[1])
    e_vector[1,0] = e_vector[1,0]/math.sqrt(e_vector[0,0]**2+1)
    e_vector[0,0] = e_vector[0,0]/math.sqrt(e_vector[0,0]**2+1)
    e_vector[1,1] = e_vector[1,1]/math.sqrt(e_vector[0,1]**2+1)
    e_vector[0,1] = e_vector[0,1]/math.sqrt(e_vector[0,1]**2+1)
    return e_value, e_vector

# Read the data file
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

# To open the data from the pickle file
with open(s,'rb') as f:
    data = pickle.load(f)

# To place the data in x and y and to plot the data
x,y = zip(*data)
matrix = np.zeros((len(x), 2))
for i in range(len(x)):
    matrix[i, 0] = x[i]
    matrix[i, 1] = y[i]

#Find the covariance matrix using the given data
cov_mat = covMat(matrix.T)

# Compute the eigenvalues and eigenvectors using covariance matrix
e_value = np.ones(2)
e_vector = np.ones((2,2))

x,y = eigen_values(cov_mat)
print("Eigenvalues are:",x)
print("EIgenvectors are:",y)
y = np.array([[x[0]*y[0,0],x[1]*y[0,1]],[x[0]*y[1,0],x[1]*y[1,1]]])/100

y =y.T

#Plot the obtained eigenvectors usign eigen-decomposition
origin = [0], [0]
plt.scatter(*zip(*data))
plt.quiver( *origin, y[:,0], y[:,1], color =['r','g'], scale = 90, width=0.003)
#plt.axis([-150,1500,-1000,1000])

plt.show()
