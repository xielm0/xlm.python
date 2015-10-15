# -*- coding: utf-8 -*-
"""

@author: xlm
""" 

import numpy as np
import random
import matplotlib.pyplot as plt

all_points=[]   #所有点
core_points=[]  #核心点  (x,y) 
 
#a,b两点距离计算
def distance(a,b):
    d=0
    for i in xrange(0,2):
        d+=pow(a[i]-b[i],2)
    return d**0.5
 

#随机创建100个点,
rand_points=np.random.randint(1,50,(100,2)) #size=(100,2) 
rand_points=rand_points.tolist()

#去重
all_points=[]
for point in rand_points:
    if point not in all_points:
        all_points.append(point)




    
 
      
        
        
 
    
