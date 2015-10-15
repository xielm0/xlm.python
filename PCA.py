#!/usr/bin/env python
# -*- coding: utf-8 -*-
#code:myhaspl@qq.com
#7-22.py 
import numpy as np
import matplotlib.pyplot as plt
import mlpy
np.random.seed(0)
mean, cov, n = [0, 0], [[1,1],[1,1.5]], 100
x = np.random.multivariate_normal(mean, cov, n)
pca = mlpy.PCA()
pca.learn(x)
coeff = pca.coeff()
fig = plt.figure(1) 
plot1 = plt.plot(x[:, 0], x[:, 1], 'o')
plot2 = plt.plot([0,coeff[0, 0]], [0, coeff[1, 0]], linewidth=4, color='r')
plot3 = plt.plot([0,coeff[0, 1]], [0, coeff[1, 1]], linewidth=4, color='g') 
xx = plt.xlim(-4, 4)
yy = plt.ylim(-4, 4)
z = pca.transform(x, k=1) 
xnew = pca.transform_inv(z) 
fig2 = plt.figure(2)
plot1 = plt.plot(xnew[:, 0], xnew[:, 1], 'o')
xx = plt.xlim(-4, 4)
yy = plt.ylim(-4, 4)
plt.show()