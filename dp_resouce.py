# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:45:22 2014
设某工厂有 1000 台机器，生产两种产品 A、B，若投入x台机器生产 A产品，则纯收入为5x ，若投入 y 台机器生产B种产品，则纯收入为4y，
又知：生产 A种产品机器的年折损率为20%，生产B 产品机器的年折损率为10%，问在5 年内如何安排各年度的生产计划，才能使总收入最高？


这个题目，只能逆序解法，找不到前推发，因为写不出顺序情况下的转移方程。
逆序解法，可以这样定义：f(k,x)表是x台机器在k年的最大收入。
转移方程，f(k-1,x)=k-1年产值+f(k,k-1年剩余机器数)，可以递归。
而前推法，k年的最大收入=前k-1年最大收入+k年产值 这是不成立的。因为k年的产值依赖剩余机器数。前k-1年的最大收对计算k年的最大收入没有直接帮助。



@author: xlm
""" 

def dp_opt2(k,x,n,tmp):
    '''动态规划，将中间结果存储起来。
    k表示年，x表示第k年剩余的机器,n,总共几年计划
    转移方程  v[k]=max{5u+4(x-u)+v[k+1]}
    '''
    global z
    try: 
        return tmp[(k,x)]     
    except:    
        if k==n+1:
            return 0
        res=[0 for i in range(x+1)]
        for u in range(x+1): #100台机器有101个选择
            m=int(0.8*u+0.9*(x-u))
            res[u]=u+4*x+dp_opt2(k+1,m,n,tmp)   
            
        tmp[(k,x)]=max(res) 
        z= res.index(max(res))
        return max(res) 

def dp_opt(x,n):
    global tmp
    tmp={}
    return dp_opt2(1,x,n,tmp)
    
def show(x,n):
    global z
    m=x
    for k in range(1,n+1)[::-1]: 
        result=dp_opt(m,k)
        print u'第', n+1-k, u'年后最大总收入=' ,result
        g=m
        m=int(0.8*z+0.9*(m-z)) #剩余机器台数
        print u'生产A,B机器分配为',z,g-z,u'年后剩余',m,u'台'
        

result=dp_opt(1000,5)
print u'最大总收入=',result 
show(1000,5) 

    
 
      
        
        
 
    
