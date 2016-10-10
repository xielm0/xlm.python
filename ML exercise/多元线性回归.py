#!/usr/bin/env python
# -*- coding: utf-8 -*-
#code:xlm
#bx=y 
import numpy as np

x =np.matrix([[7,2,3],[3,7,17],[11,3,5],[8, 4,5]],dtype=np.float64)
y =np.matrix([28,40,44,30],dtype=np.float64).T
a=(x.T*x).I*x.T*y 
#a=np.linalg.solve(x,y)
temp_e=y-x*b
mye=temp_e.sum()/temp_e.size
e=np.matrix([mye,mye,mye]).T 
print 'a=',a 
print 'x=',x
print 'y=',y
print "y=%f*x1+%f*x2+%f*x3+%f"%(a[0],a[1],a[2],mye)
print u'模拟计算y=',x*b

 
plt.plot(x,y)
plt.show()