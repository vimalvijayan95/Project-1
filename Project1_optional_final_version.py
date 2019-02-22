# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 00:09:02 2019

@author: andrea
"""
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Load the terrain
terrain1 = imread('SRTM_data_Madrid.tif')


# Show the terrain
plt.figure()
plt.title('Terrain over Madrid')
plt.imshow(terrain1, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

y = np.arange(0, len(terrain1))
x = np.arange(0, len(terrain1))
x, y = np.meshgrid(x,y)

#Plotting the 3D figure
real = plt.figure()
ax = real.gca(projection='3d')
ax.set_title('Real data')
surf = ax.plot_surface(x, y, terrain1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.set_xlabel('Position $x$')
ax.set_ylabel('Position $y$')
ax.set_zlabel("Terrain Height")
plt.show()

# OLS regression 

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

poly = PolynomialFeatures(degree=9) 
X = poly.fit_transform(np.c_[x.ravel(), y.ravel()]) # Design Matrix
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(terrain1.ravel()) #beta to calculate the solution
zfit = X.dot(beta).reshape(len(terrain1),len(terrain1))

#clf2 = LinearRegression()
#clf2.fit(XY, terrain) - automatic calculation using sklearn crashes in the computer
#zfit = clf2.predict(XY)


#plot of the fitted polynomial
fit = plt.figure()
ax = fit.gca(projection='3d')
ax.set_title('Fitted data with polynomial')
surfit = ax.plot_surface(x, y, zfit, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlabel('Position $x$')
ax.set_ylabel('Position $y$')
ax.set_zlabel("Terrain Height")
plt.show()

#MSE and R2 score
mse=mean_squared_error(terrain1, zfit)
print("MSE with OLS no resampling",mse)
r2=r2_score(terrain1, zfit)
print("R2 score with OLS no resampling",r2)


# Ridge without resampling

lmb_values = [1e-4, 1e-3, 1e-2, 10, 1e2, 1e4]

for lam in lmb_values:
    betaRidge = np.linalg.inv(X.T.dot(X)+ lam*np.identity(X.shape[1])).dot(X.T).dot(terrain1.ravel())
    
    fitRidge=X.dot(betaRidge).reshape(len(terrain1),len(terrain1))
    
    mseRidge=mean_squared_error(terrain1, fitRidge)
    r2Ridge=r2_score(terrain1, fitRidge)
    
    #MSE and R2
    print("R2 score for Ridge regression without resampling:", r2Ridge)
    print("MSE for Ridge regression without resampling:", mseRidge)

