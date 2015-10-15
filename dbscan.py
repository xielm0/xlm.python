# -*- coding: utf-8 -*-
''' 
# Purpose:     密度聚类算法
# 1, 任意选择一个点，找出半径r之内的点并计算个数n ,如果n>=最小阀值minpts,则形成一个簇，该点被标志为已访问点；如果<阀值，该点被标志为噪声；
# 2，簇扩展。以相同的方法处理簇内其他未被标志为“已访问”的点。如果n2>n1,则新簇取代旧簇，该点是新的核心点。
# 3，以同样的方法处理簇外的点，形成新簇
# Author:      xlm
'''
import numpy as np
import random
import matplotlib.pyplot as plt

all_points=[]   #所有点
core_points=[]  #核心点  (x,y)
plotted_points=[]#标记已访问的点  （x,y）
unplotted_points=[]#未访问的点
cluster_points=[]  #簇点 （x,y,簇类）如（1，2，1）
noise_points=[] #噪声

r=5    #半径
minPts=5  #簇最小个数
cluster_tag=0     #簇的标签
n_repeats=0       #循环次数

#a,b两点距离计算
def distance(a,b):
    d=0
    for i in xrange(0,2):
        d+=pow(a[i]-b[i],2)
    return d**0.5

#计算半径内的个数，并返回半径内的点 （包括核心点）
def point_nums(r,i_point,i_points):
    total=0
    r_points=[]
    for point2 in i_points:
        d=distance(i_point,point2)
        if d<=r:
            total+=1
            r_points.append(point2)
    return total,r_points

#随机创建100个点,
rand_points=np.random.randint(1,50,(100,2)) #size=(100,2) 
rand_points=rand_points.tolist()

#去重
all_points=[]
for point in rand_points:
    if point not in all_points:
        all_points.append(point)

unplotted_points=all_points[:]   #未访问的点
plotted_points=[]  #已经访问的点

#开始训练
while len(unplotted_points)>0:
    n = 0
    tmp_points=[]

    #找到核心点和簇内的点
    for point in unplotted_points:

        n,tmp_points=point_nums(r,point,unplotted_points)  #计算半径内的点个数，新簇从未访问的点中选择
        if n >= minPts:
            #将半径内的点加入簇
            core_point=point[:]
            cluster_tag+=1  #簇标志，找到一个核心点则创建一个簇

            for point2 in tmp_points:
                point22=point2[:]
                point22.append(cluster_tag)   #增加一列，2列变3列
                if point2 not in plotted_points :
                    cluster_points.append(point22)  #它的格式是：（x,y,cluster_tag）
                    plotted_points.append(point2)    #标志已经访问，格式是 (x,y)

            break     #找到一个簇，则下一步进行簇扩展
        else:
            pass


    #如果没有新簇增加则结束
    #循环一次，则增加一个新簇，如果没有增加新簇，则循环次数大于簇类
    n_repeats+=1
    if cluster_tag<n_repeats:
        break

    #簇扩展
    cluster=[x[0:2] for x in cluster_points if x[2]==cluster_tag]  #左开右闭 ，x[0:2]前2列
    a=len(cluster)

    i=0
    while i<a :
        point=cluster[i]  #point的格式为[50,100]

        try:
            unplotted_points.remove(point)
        except:
            pass
        if point==core_point: #核心点不用再扩展了
            i+=1
            continue
        n,tmp_points=point_nums(r,point,all_points)  #判断是否满足条件，条件判定从全部点中选择
        if n>=minPts: #将半径内的点加入cluster
            for point2 in tmp_points:
                point22=point2[:]
                point22.append(cluster_tag)
                if point2 not in plotted_points :
                    cluster_points.append(point22)
                    plotted_points.append(point2)
        i+=1
        cluster=[x[0:2] for x in cluster_points if x[2]==cluster_tag]
        a=len(cluster) # cluster在扩展，长度len也就在动态增加

#画图
markers = ['ok','^r','*r','db','vb','>c','<k','+r','.b','p']
plt.axis((0,60,0,60))

#聚类点
for ii in range(len(cluster_points)):
    cluster=cluster_points[ii]
    i=cluster[2]
    i=(i-1)%10  #只有10个形状，所以hash求模
    plt.plot(cluster[0],cluster[1],markers[i])


#噪声点=all_points-已经访问的点
noise_points=[]
for point in all_points:
    if point not in plotted_points:
        noise_points.append(point)

for ii in range(len(noise_points)):
    plt.plot(noise_points[ii][0],noise_points[ii][1],'.y')

plt.show()
