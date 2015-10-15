#!/usr/bin/env python
# -*- coding: utf-8 -*-
#code:myhaspl@qq.com
#7-19.py 
#y=b1*x+b2*(x^2)+b3*(x^3)
import numpy as np
import matplotlib.pyplot as plt
z=np.matrix([3,1.4,1.9]).T
myx =np.matrix([[7],[3],[9]],dtype=np.float64)
x = np.matrix([[myx[0,0],myx[0,0]**2,myx[0,0]**3],\
               [myx[1,0],myx[1,0]**2,myx[1,0]**3],\
               [myx[2,0],myx[2,0]**2,myx[2,0]**3]],\
               dtype=np.float64)
y =x*z
b=(x.T*x).I*x.T*y
print y
print u"参数项矩阵为{0}".format(b)
i=0
cb=[]
while  i<3:
    cb.append(b[i,0])
    i+=1
temp_e=y-x*b
mye=temp_e.sum()/temp_e.size
e=np.matrix([mye,mye,mye]).T
print cb
print "y=%f*x+%f*x^2+%f*x^3+%f"%(b[0],b[1],b[2],mye)

pltx=np.linspace(0,10,1000)
plty=cb[0]*pltx+cb[1]*(pltx**2)+cb[2]*(pltx**3)+mye
plt.plot(myx,y,"*")
plt.plot(pltx,plty)
plt.show()