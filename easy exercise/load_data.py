# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Name:        模块1
# Purpose:
#
# Author:      xlm
#
# Created:     05/03/2014
# Copyright:   (c) xlm 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------


import math
import sys

def load_data(fname):
        """
                从文件加载训练或测试数据
                返回得到的输入矩阵x，输出向量y，
                以及x各维特征的最大值x_max，y的最大值y_max（用于确定x,y的取值范围，它们都是离散值）
        """
        f = open(fname)
        y=[]
        x=[]
        for line in f: #类似line=f.readlines()
                str_list=line.strip('\n').split(' ')  #trim 
                xi=[]
                for v in str_list:  #v=c5 , f20
                        if v[0] == 'c':
                                yi=v[1:]
                                y.append(int(yi))
                        if v[0] == 'f':
                                xij=v[1:]
                                xi.append(int(xij))
                x.append(xi)  #x是矩阵
        f.close()
        y_max=1
        for yval in y:
                if yval > y_max:
                        y_max=yval
        x_max=[]
        for a in x[0]:
                x_max.append(a)
        for b in x:
                for i in range(len(b)):
                        if b[i] > x_max[i]:
                                x_max[i] = b[i]
        return x,y,x_max,y_max



x_matrix,y_list,x_max_list,y_max=load_data("E:\python\python练习\max_ent\zoo.train")