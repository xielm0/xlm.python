# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 15:30:23 2014

@author: xlm
"""
import numpy as np
import matplotlib.pyplot as plt 
import neurolab as nl 
x1=np.arange(0,10,0.1)
y1=sin(x1)
y2=cos(x1) 
train_x=[]
d=[]  #d =[[1],[0],[1],[0],[1],[0]]
for i in xrange(0,len(x1)):
    train_x.append([x1[i],y1[i]])
    d.append([1])
    train_x.append([x1[i],y2[i]])
    d.append([0])
     
input=np.array(train_x)
target=np.array(d)
#newff与matlab一样，生成一个前馈BP网络
#newff(minmax, size, transf=None）
net=nl.net.newff([[0,20],[-1,1]],[8,8,1])  #size=[8,1]
err=net.train(input,target,epochs=2000, show=50, goal=0.01)
out=net.sim(input)  
#画图
mymean=np.mean(out)  
x_max=np.max(input[:,0])+5  
x_min=np.min(input[:,0])-5  
y_max=np.max(input[:,1])+5  
y_min=np.min(input[:,1])-5  
  
#误差曲线 
plt.subplot(211)
plt.plot(range(len(err)),err)    
  
#可视化图 
plt.subplot(212)
plt.xlim(x_min,x_max)  
plt.ylim(y_min,y_max)  
for i in xrange(0,len(input)):  
    if out[i]>0.5:  
        plt.plot(input[i,0],input[i,1],'ro')  
    else:  
        plt.plot(input[i,0],input[i,1],'g*')  
  
plt.show()  