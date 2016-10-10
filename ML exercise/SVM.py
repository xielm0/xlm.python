# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:45:22 2014

@author: xlm
"""
import numpy as np
import matplotlib.pyplot as plt

#核函数计算,目前只有rbf函数
def kernel(x,xi,gama=1): 
    dif =x-xi
    ss=np.inner(dif,dif)
    v=np.exp(-gama*ss)
    return v
    

# 初始化一些值
class libsvm:  
    def __init__(self, dataSet, labels, C, gama):  
        self.train_x = np.array(dataSet) # each row stands for a sample  
        self.train_y = np.array(labels)  # corresponding labe
        self.C = C             # slack variable  
        self.try_max = toler     # termination condition for iteration  
        self.x_len = len(train_x) # number of samples  
        self.alphas = matrix(zeros((self.x_len, 1))) # Lagrange factors for all samples  
        self.b = 0  
        self.e_cache = matrix(zeros((self.x_len, 2)))  
        #self.kernelMat = calcKernelMatrix(self.train_x, self.kernelOpt)
        



def calc_w(x,y,alphas):
    '''
    w=sum(alpha*yi*xi) 
    xi=np.array([1,2]),yi=np.array([1]),alpha=np.array([0.3])     
    wi=np.multiply(xi,yi,alphas[i]) 
    ''' 
    w=np.zeros(x[1])    
    for i in xrange(x_len):
        w+=x[i]*y[i]*alphas[i]        
    return w
 
def pred_y(w,x,b):
    y=np.inner(w.T*x)-b
    return y
    
def calc_e(x,y,w,b):
    pred_y=pred_y(w,x,b)
    e=y-pred_y
    return e
    
def get_e_array(x,y,w,b):
    '''得到所有的e,存放在数组中'''
    global e_cache
    for i in xrange(x_len):
        e_cache=calc_e(x[i],y[i],w,b)


def select_lambda():
    '''1，选取违背KKT条件最严重的点作为a1'''
    target_i=None
    target_j=None
    if target_i==None :
        for i in xrange(x_len):
            if np.inner(w.T*x)-b>1 and alpha[i]<>0:
                target_i=i
    if target_i==None :
        for i in xrange(x_len):
            if np.inner(w.T*x)-b==1 and (alpha[i]==0 or alpha[i]==C):
                target_i=i
    if target_i==None :
        for i in xrange(x_len):
            if np.inner(w.T*x)-b<1 and alpha[i]<>C:
                target_i=i        
    #接下来找到j,要求|ei-ej|最大
    max_dif=0
    for j in xrange(x_len):
        dif=abs(e_cache[target_i]-e_cache[j])
        if dif>max_dif:
            max_dif=dif
            target_j=j 
            
    return target_i,target_j

def get_eta(i,j):
    '''eta=2k12-k11-k22'''
    eta=2*kernel(x[i],x[j])-kernel(x[i],x[i])-kernel(x[j],x[j])
    return eta


def learn():
    '''
    SMO算法
    1,选取一个违背KKT条件的点x1
    2,根据|E1-E2|,选取满足最大值的x2
    3,计算eta,计算a2的上下界
    4，计算a2的new值，并将a2的新值卡定在上下界内。
    5，更新a1值。
    6，计算w值，
    7，计算新的b值
    '''  
    i,j=select_lambda()
             