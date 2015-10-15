# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 18:21:50 2015

@author: xlm
"""


def wc(file_name):
    data = open(file_name,'r')  
    n=0    
    while 1: 
        line = data.readline()        
        if not line: 
            break; 
        n=n+1    
    data.close()
    return n
    
m=wc(u'E:\广州BI\临时需求\SP and discnt\科大讯飞\科大订购_201407.csv')
print m