# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 15:08:02 2014

@author: xlm
"""
import numpy as np
import random
import urllib
import time 

#last_red=[7,15,16,25,28,32]
#last_blue=[5]
#从网上获取中奖号码的内容
url='http://www.17500.cn/getData/ssq.TXT'
page=urllib.urlopen(url)
data=page.read()
page.close()

#将历史中奖号码保存下来，写文件
path='E:\\R\\exercise' 
filename= 'red_blue.txt'
filename=path+'\\'+filename  
f=open(filename,'w')
f.write(data)  
f.close() 

#根据上期的中奖号码预测中奖号码
#得到上期中奖号码，并转化为数字
data_list=data.split('\n')  #按换行符分割，变成数组
#倒数第2行为最新的中奖号码,因为最后一行为空
tmp=data_list[-2]  #2015012 2015-01-27 03 05 22 23 29 31 06
tmp=tmp.split(' ')  
last_red1=tmp[2:8]  #['03', '05', '22', '23', '29', '31']
    
last_red=str2int(last_red1) #转换为数字类型
last_blue=[]
last_blue.append(eval(tmp[8]))
print u'上期中奖号码：',last_red,last_blue

#将['03', '05']转化[3,5]
def str2int(str_list):
    int_result=[]
    for num in str_list:
        int_result.append(eval(num))
    return int_result

#将int数组转化成字符串，便于写入数据文件
#sep.join() 不能对int的list进行
def int2str(int_list,sep):
    str_result=''
    for a in int_list:
        a="%02d"%(a)       #位数不足前面补0
        str_result+=str(a)+sep 
    
    str_result=str_result[0:-1]  # 最后一个分隔符不要  
    return str_result
    
   
#将预测结果写入文件
path='E:\\R\\exercise'
day=time.strftime('%Y%m%d',time.localtime( )) #20150204
filename=day+'.txt'
filename=path+'\\'+filename
sep ='\t'
print filename
f=open(filename,'w')

base_red=range(1,33,1) #1-33
base_blue=range(1,16,1) #1-16

print u'本期预测10个中奖号码，如下：'

n=0
i=0
while n<10000  :
    n+=1
    red=random.sample(base_red,6)
    blue=random.choice(base_blue)
    red.sort()  #一定要加括号
    flag1=0
    flag2=0
    c1=0
    c2=0
    c3=0
    b=-1
    for a in red :
        if a in last_red:  #有1个号码在上次的号码中
            flag1=1
        
        if a==b+1:  #连号
            flag2=1
        b=a
        #33个号码分3个区，每个区至少有1个号码
        if a<=11 :
            c1=1
        elif a<=22 :
            c2=1
        else: 
            c3=1
    #打印前10个号码，其他的写入文件
    if flag1==1 and flag2==1 and c1+c2+c3==3 :            
        f.write(int2str(red,sep)+ sep + str(blue))
        f.write('\n')
        i+=1 
    
        if i<=10:        
            print '%d:'%(i),red,blue 
        elif i>=1000:
            break  
    
f.close()

#预测准确率验证
    