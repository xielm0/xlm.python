# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:45:22 2014
dp-背包问题  -- 前推法
有编号分别为a,b,c,d,e的5件物品，它们的重量分别是2,2,6,5,4，它们的价值分别是6,3,5,4,6
现在给你个承重为10的背包，如何让背包里装入的物品具有最大的价值总和？

res[i][j] 记录在最大承重为j，可选物品为前i个物品的最大价值
转移方程： res[i][j]=max{res[i-1][j],res[i-1][j-wi]+v[j]}
res[i-1][j-wi]+v[j] 表示假设最大承重j减去第i个物品重量后的值作为最大承重，可选物品为前i-1个物品下的最大价值 + 第i个物品的价值。
当然要求j>wi
@author: xlm
""" 
def dp_bag(w,v,i,aw,m):
    '''0/1背包问题,动态规划。递归方程
    用字典m存储中间结果 
    w：weight   v：value    i：物品个数    aw：allowable weight背包承受重量 
    
    状态转移：最大价值=max｛承重c可选前i-1个物品的最大价值，承重c可选前i个物品的最大价值｝
    承重c可选前i个物品的最大价值=承重c-w[i]可选前i-1个物品的最大价值+第i个物品的最大价值。 --这个才是真正的转移方程。
    '''  
    try:  
         return m[(i, aw)]
    #如果m[(i, aw)]已经存在，则直接取，否则计算一次。
    except KeyError:
        if aw<=0:
            return 0
        #第1件物品的判断    
        if i == 0:
            if w[i] > aw:
                return 0
            else :
                return v[0]
         
        without_i = dp_bag(w,v,i-1,aw,m)  
        if w[i] > aw:
            m[(i,aw)] = without_i  
            return without_i
        else:
            with_i = v[i] + dp_bag(w,v,i-1,aw-w[i],m)
            m[(i,aw)] = max(without_i,with_i) 
            return max(without_i,with_i)

def dp_bag2(w,v,i,aw):
    global m
    m = {}
    return dp_bag(w,v,i,aw,m)
    
##Test example
w = [2,2,6,5,4]
v = [6,3,5,10,6]  
res = dp_bag2(w, v, len(v)-1,7) #python的索引是从0开始
print u'最大价值为:',res  


############################################################
def bag(w,v,n,c):
    '''0/1背包问题,动态规划。 
    w：weight   v：value    n：物品个数    c： 背包承受重量 
    ''' 
    #创建一个初始列表,多1个初始列，值为0
    res=[[0 for j in range(c+1)] for i in range(n+1)]  
    
    #状态转移
    #
    for i in range(1,n+1):
        for j in range(1,c+1): 
            without_i=res[i-1][j]
            res[i][j]=without_i
            if w[i-1]>j: 
                res[i][j]=without_i
            else :
                with_i=res[i-1][j-w[i-1]]+v[i-1]
                res[i][j]=max(without_i,with_i) 
    return res

def show(w,res,n,c):
    print u"最大价值为:",res[n][c] 
    x=[False for i in range(n)] 
    j=c
    #逆序，选择改变的最大值的物品
    for i in range(1,n+1)[::-1]:
        if res[i][j]>res[i-1][j]: 
            x[i-1]=True
            j-=w[i-1]
    print u'选择的物品为:' 
    for i in range(n):
        if x[i]:
            print u'第',i+1,u'个'  
 
w=[2,2,6,5,4]
v=[6,3,5,10,6]
res=bag(w,v,5,7)
show(w,res,5,7)   
