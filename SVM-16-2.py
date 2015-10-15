#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mlpy 

#数据准备
x1=np.arange(0,10,0.1)
y1=sin(x1)
y2=cos(x1)
train_x=[]
d=[]   
for i in xrange(0,len(x1)):
    train_x.append([x1[i],y1[i]])
    d.append(1)
    train_x.append([x1[i],y2[i]])
    d.append(0)
     
train_x=np.array(train_x) 
target=np.array(d) 
#训练
svm = mlpy.LibSvm(kernel_type='rbf', gamma=10)
#svm = mlpy.LibSvm(kernel_type='poly', gamma=10,degree=3)
svm.learn(train_x,target) 
#仿真预测
out=svm.pred(train_x) 
#画图    
x_max=np.max(train_x[:,0])+1  
x_min=np.min(train_x[:,0])-1  
y_max=np.max(train_x[:,1])+2  
y_min=np.min(train_x[:,1])-2 
#可视化图  
plt.subplot(111) 
plt.xlim(x_min,x_max)  
plt.ylim(y_min,y_max)  
for i in xrange(0,len(train_x)):  
    if out[i]>0.5:  
        plt.plot(train_x[i,0],train_x[i,1],'ro')  
    else:  
        plt.plot(train_x[i,0],train_x[i,1],'g*')  
  
plt.show() 