# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:45:22 2014
设某工厂有 1000 台机器，生产两种产品 A、B，若投入x台机器生产 A产品，则纯收入为5x ，若投入 y 台机器生产B种产品，则纯收入为4y，
又知：生产 A种产品机器的年折损率为20%，生产B 产品机器的年折损率为10%，问在5 年内如何安排各年度的生产计划，才能使总收入最高？

阶段：第1年，第2年，第3年，第4年，第5年
状态：总收入，剩余n台机器，(1000种状态)
决策：u生产A的机器x台，生产B的机器n-x台，1000个决策
重点来了：决策后会产生状态。那么状态转移方程是：
状态转移：f(总收入|剩余n台机器)=max(f(总收入1|剩余n台机器), f(总收入2|剩余n台机器))

@author: xlm
""" 
import copy

def dp_opt(x):
    """状态。x[1]=[剩余机器台数,总收入],优化下： x[剩余机器台数]=总收入""" 
    y=copy.deepcopy(x)
    #决策
    #遍历每一种状态 
    for i,xi in enumerate(x):
        #i为剩余机器数 ,有i+1中选择
        for j in xrange(i+1):
            v=5*j + 4*(i-j)
            #剩余机器数
            tmp=int(0.8*j)+int(0.9*(i-j)) 
            #状态转移
            y[tmp]=max(y[tmp],v+xi)    
            
    return y

def main():
    #初始化
    print 'start '
    x=np.zeros(1001)
    #按年分阶段
    for k in xrange(5):
        y=dp_opt(x) 
        x=y
    print u'最大总收入=',max(y)
    
main()

################################################
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

def dp_opt3(x,n):
    global tmp
    tmp={}
    return dp_opt2(1,x,n,tmp)
    
def show(x,n):
    global z
    m=x
    for k in range(1,n+1)[::-1]: 
        result=dp_opt3(m,k)
        print u'第', n+1-k, u'年后最大总收入=' ,result
        g=m
        m=int(0.8*z+0.9*(m-z)) #剩余机器台数
        print u'生产A,B机器分配为',z,g-z,u'年后剩余',m,u'台'
        

result=dp_opt3(1000,5)
print u'最大总收入=',result 
show(1000,5) 


    
 
      
        
        
 
    
