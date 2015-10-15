# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 16:09:27 2014
1000的阶乘的结果很大,没有哪个数字类型能存储。解决方式就是通过数组来实现。最后结果通过数组拼接。
例如：123445566*8= 123 445 566 *8 =566*8 +445*8 + 123*8
@author: xlm
"""
import numpy as np

#定义一个数组，用于存放结果
#数组大小
#我们知道乘以10，位数增加1；乘以100，位数增加2，长度=log1+log2+log3+...+log(n)
N=1000
n=reduce(lambda x,y:x+log10(y),xrange(1,N+1))
n=int(n//3+1)
print n
#
a=np.zeros((n),int)
a[0]=1
#阶乘运算
for i in xrange(2,N+1):
    jin=0
    for j in xrange(0,n):
        temp=a[j]*i+jin
        a[j]=mod(temp,1000)
        jin=temp//1000 #进位

#打印结果
for i in xrange(n-1,-1,-1):
    #如果a[i]=0,打印3个0
    if a[i]==0:
        print '000',
    else:
        print a[i],        