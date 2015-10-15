#!/usr/bin/env python
#-*- coding: utf-8 -*-
#create_user:xlm
#模拟的是c+ax+by=0的直线 ,转化为线性方程就是向量v(1,x,y) ,权重w(c,a,b) ，向量v*权重w即是线性方程组。
import numpy as np
import matplotlib.pyplot as plt
import math

c=1.0      #偏置
a0=0.9    #学习率 
r=50       #退火算法中的因子,
expect_e=0.05   #误差阀值
max_count=500   #最大训练次数

w=np.array([c,0,0])  #初始权重
#训练样本
traindata1=np.array([[9,25,-1],[5,8,-1],[15,31,-1],[35,62,-1],[19,40,-1],[28,65,1],[20,59,1],[9,41,1],[12,60,1],[2,37,1]])
#traindata1=np.array([[1,6,1],[2,12,1],[3,9,-1],[8,24,-1]])
x=[]
d=[]
n=len(traindata1[0])
for xn in traindata1:
    x.append([1]+xn[0:n-1].tolist())  #需要在前面加一个1.，因为向量为(1,x,y)
    d.append(xn[n-1].tolist())  #最后一列为分类标志
x=np.array(x)
d=np.array(d)

#函数
def sgn(y):
        if y>0:
                return 1
        else:
                return -1
def get_v(i_w,i_x):
        return sgn(np.dot(i_w.T,i_x))
def get_e(i_w,i_x,i_d):
        return i_d-get_v(i_w,i_x)
#权重w训练方法
def neww(oldw,i_d,i_x,i_a):
        e=get_e(oldw,i_x,i_d)
        new_w=oldw+i_a*e*i_x   #梯度下降
        return new_w,e
#开始训练
mycount=0
while True:
        mye=0
        a=a0/(1+float(mycount)/r)  #模拟退火
        i=0
        for xn in x:
                w,e=neww(w,d[i],xn,a)
                #print w,e
                i+=1
                mye+=pow(e,2)
        mye/=float(i)       #方差，最小均方
        #mye=math.sqrt(mye)
        mycount+=1
        print u"第%d次调整后的误差：%f ,权值"%(mycount,mye) ,
        print w
        #结束
        if mye<expect_e or mycount>max_count:break

for xn in x:
        print "%d , %d => %d "%(xn[1],xn[2],get_v(w,xn))

#画图
plot=x
x_max=max(plot[:,1])+5.0   #[:,1]必须是array类型才能用，list不可以。
x_min=min(plot[:,1])-5
y_max=max(plot[:,2])+5.0
y_min=min(plot[:,2])-5
plt.axis((x_min,x_max,y_min,y_max))
for i in range(len(x)):
    if d[i] > 0:
        plt.plot(x[i][1],x[i][2],'or')
    else:
        plt.plot(x[i][1],x[i][2],'*b')

#模拟的是c+ax+by=0的直线 ,转化为线性方程就是向量v(1,x,y) ,权重w(c,a,b) ，向量v*权重w即是线性方程组。
#虚线，y=(-c-ax)/b
c= w[0]
a= w[1]
b= w[2]
y1=(-c-a*x_min)/b
y2=(-c-a*x_max)/b
plt.plot([x_min, x_max], [y1,y2], 'b--')

plt.show()
