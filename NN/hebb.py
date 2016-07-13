# -*- coding: utf-8 -*-
"""
Created on Fri Jul 08 18:11:58 2016

@author: xieliming
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul 08 16:52:52 2016

@author: xieliming
#只适合线性
"""
import numpy as np 
a=0.1 #学习率 
x = np.array([[1,1],[2,1],[1,-1],[-1,-2]])
d = np.array([1,1,-1,-1])
w=np.array([1,0 ])
def sign(v):
        if v>0:
                return 1 
        else:
                return -1
                
def comy(myw,myx):
        return sign(np.dot(myw.T,myx))
def hebb(oldw,myd,myx,a): 
        w= oldw + a* comy(oldw,myx)*myx  
        return w

def update_w(w,x,a):      
    e=0
    for i in range(0,len(x)):
            w=hebb(w,d[i],x[i],a) 
            if comy(w,x[i]) <> d[i]:
                e +=1
    return (w,e)        

i=0  
while True:
    w,sum_e =update_w(w,x,a)   
    print w        
    i+=1
    if i>=10 or sum_e<=1: break
    

        
z= comy(w,x[1])
print z

