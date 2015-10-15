#!/usr/bin/env python
# -*- coding: utf-8 -*-
#code:myhaspl@qq.com
#7-17.py
 
import matplotlib.pyplot as plt
x =[1,2,3,3,6,12,11]
y =[3,5,8,5,12,26,20]
x=[0,1,2,3,4,5,6,7]
y=[1.4,1.3,1.4,1.1,1.3,1.8,1.6,2.3]
average_x=float(sum(x))/len(x)
average_y=float(sum(y))/len(y)
#x-x均值
x_sub=map((lambda x:x-average_x),x) 
y_sub=map((lambda y:y-average_y),y)
#
x_sub_pow2=map((lambda x:x**2),x_sub)
y_sub_pow2=map((lambda x:x**2),y_sub)
#（x-x均值）
x_y=map((lambda x,y:x*y),x_sub,y_sub) 
#------自带计算，类回归系数计算-------------------------
#sum(x_y)=cov(x,y)*n
#sum(x_sub_pow2) =var(x)*n
a=float(sum(x_y))/sum(x_sub_pow2) 
b=average_y-a*average_x
print 'line1=%f *x+%f'%(a,b)
#---------回归系数，标准计算--------------------------
#cov(x,y)
cov_xy=np.mean(x_y)
#r=cov(x,y)/(sd(x)*sd(y)) ;a=r*sd(y)/sd(x)=cov(x,y)/var(x)
var_x=np.var(x)
#数据有差异，是小数计算的问题。比较a值我们发现，上面的计算更准确。通过误差比较也可以验证
print 'a=',cov_xy/var_x 
#-----最小二乘法，标准计算---------------------------
x_y=map((lambda x,y:x*y),x,y)  #x*y
S_xy=sum(x_y)-sum(x)*sum(y)/len(x)
x_pow2=map((lambda x:x**2),x)
S_xx=sum(x_pow2)-sum(x)*sum(x)/len(x)
a2=S_xy/S_xx
b2=average_y-a2*average_x
print 'line2=%f *x+%f'%(a2,b2)

#----画图--------------------- 
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(x, y, '*')
plt.plot([0,15],[0*a+b,15*a+b],'r')
plt.plot([0,15],[0*a2+b2,15*a2+b2],'g')
#plt.grid()
#plt.title("{0}*x+{1}".format(a,b))
plt.show()
#----误差比较-----------------
p_y=[]
p_y2=[]
mse1=0
mse2=0
for i in xrange(0,len(x)):
    y1=a*x[i]+b
    p_y.append(y1)
    y2=a2*x[i]+b2
    p_y2.append(y2)
    e1=(y1-y[i])**2
    e2=(y2-y[i])**2
    mse1+=e1
    mse2+=e2
print mse1,mse2  