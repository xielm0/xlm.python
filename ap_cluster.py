# -*- coding: utf-8 -*-
"""

@author: xlm
""" 

import numpy as np
import random
import matplotlib.pyplot as plt
import copy

maxtry=1000
eta=0.9

#训练样本
n=20
x=np.random.randint(1,50,(n,2))
x=x.tolist()  # 
print x  

#a,b两点距离计算
def dist(a,b):
    d=0
    for i in xrange(0,2):
        d+=pow(a[i]-b[i],2)
    return -(d**0.5) 

#初始化s
s=zeros((n,n)) # 相似度矩阵
for i in range(n):
    for j in range(n):
        s[i,j]=dist(x[i],x[j])

p=median(s)*2
for i in range(n):
    s[i,i]=p

r=zeros((n,n))
a=zeros((n,n))
#r=copy.deepcopy(s) 
#a=copy.deepcopy(s)

#迭代
#R(i,k)=S(i,k) - max{A(i,j)+S(i,j)}(j {1,2,……,N,但j≠k}) 
#A(i,k)=min{0,R(k,k)+sum(max(0,r(j,k))) (j {1,2,……,N,但j≠i且j≠k})
num=0 
num2=0
core_before=[0]
while True:
    num+=1
    for i in range(n):
        for k in range(n): 
            if k==0:
                c=a[i,k+1:n]+s[i,k+1:n]
            elif k==n-1:
                c=a[i,0:k]+s[i,0:k]
            else:    
                c1=a[i,0:k]+s[i,0:k]
                c2=a[i,k+1:n]+s[i,k+1:n]
                c=np.concatenate((c1,c2),0)
            r[i,k]=s[i,k]-max(c)
            for j in range(n):
                d=[]
                if j<>i and j<> k: 
                    d.append(max(0,r[j,k]))
            a[i,k]=min(0,r[k,k]+sum(d))

    #聚类中心
    core_before=core[:] #复制
    core=[] 
    for i in range(n):
        z=r[i]+a[i]
        if z.max()>0:
            core.append(z.argmax())
        else:
            core.append(i)
    
    if core == core_before : num2+=1 
    if num>=maxtry or num2 == 20 :break

print core    
#画图
markers = ['ok','^r','*r','db','vb','>c','<k','+r','.b','p']
plt.axis((0,60,0,60))

for point in x: 
    #i=(i-1)%10  #只有10个形状，所以hash求模
    plt.plot(point[0],point[1],'*' )




    
 
      
        
        
 
    
