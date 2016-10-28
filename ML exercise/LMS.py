# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 19:42:02 2016

@author: xieliming
损失函数：　均方误差
通过梯度求导拟合参数theta
y=wx+b
"""

import numpy as np 
eta=0.01 #学习率
b=1 #b=x0
x = np.array([[b,1,1],[ b,2,1],[ b,1,-1],[ b,-1,-2]])
d = np.array([1,1,-1,-1])
w=np.array([0.1,0.1,0.1])


#训练n次
#GD
flag=0
if flag:
    for n in xrange(0,1000):
        y=x.dot(w)
        #loss
        err=0.5/len(d)*sum(np.square(d-y))
        print err
        #update w 
        delta=-(d-y)/len(d)
        dw = np.dot(x.T,delta)    
        w -= eta *dw 

else :
    for n in xrange(0,1000):
        for i in xrange(0,len(x)):
            y=np.dot(x[i],w)
            dw=-(d[i]-y)
            w -= eta *dw*x[i]
        
        y=x.dot(w)
        #loss
        err=0.5/len(d)*sum(np.square(d-y)) 
        print err

print w        
print  np.dot(w,x.T)
 