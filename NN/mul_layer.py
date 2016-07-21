#!/usr/bin/env python
#-*- coding: utf-8 -*-
#code:myhaspl@qq.com
#多层感知器算法
#7-12.py
'''
1，对数据进行归约化处理，使得输入值在0-1。
1，对输入的散乱数据，经过一个矩阵变成一个均匀分布的数据，但这个目前步知道怎么处理，暂时放下。
2，经实验证明，隐藏层的初始权值矩阵很重要,特别是均值，但均值越接近于0，越能得到理想的结果。而且当均值>0.008，基本没戏
3，训练数据，包括局部梯度计算，得到最终的权值矩阵
4，测试数据
在实践中，发现节点个数影响很大，但并不是越多越好。n=8的效果不错。
容易陷入局部最优化。
输入的数据在（-1，1）比（0，1）的效果好，输入数据需要打乱，这样训练效果比较好
学习率&动量参数，
学习率大，训练速度快，容易错过全局最小值，网络不稳定；学习率小，则delta_w很少，训练时间长；
动量参数，动量参数过大，则容易错过全局最小值，从小坑跳到大坑；过少，则容易陷入局部最优(掉在坑里出不来)；
'''

import numpy as np
import matplotlib.pyplot as plt
import random
import copy 

#-------------数据规约处理-------------
#param : minmax, size=[n,m] ,transf
def npminmax(x,xmin,xmax,ymin=-1,ymax=1):
    y=(ymax-ymin)*(x-xmin)/(xmax-xmin)+ymin
    return y
   
def init_x(x,minmax,ymin=-1,ymax=1):
     """x,一个样本,minmax,一个list,如[[0,20],[-1,1]]"""  
     if np.ndim(minmax)==1:  
         y=npminmax(x,minmax[0],minmax[1])
         return y
     else :    
         y=np.zeros_like(x)
         for i in xrange(0,len(x)):
             y[i]=npminmax(x[i],minmax[i][0],minmax[i][1])
         return y
         
def init_input(my_input,minmax,m):
    n=len(my_input) 
    #m > len(my_input[0])
    y=np.zeros((n,m))  
    for i,x in enumerate(my_input):
        xx=init_x(x,minmax)
        for j,a in enumerate(xx):
            y[i][j]=a 
            
    return y

def init_d(x,minmax,ymin=-1,ymax=1):
    """y的元素是数字"""
    y=x
    for i,xi in enumerate(x):
        y[i]=init_x(xi,minmax,ymin=-1,ymax=1) 
    return y  
        
def init_b(k,n):
    b=np.random.uniform(-1.0,1.0,(k,n))
    return b
    
def init_w(k,n,m):
    '''初始化w矩阵,大小kxnxn '''
    w=np.random.uniform(-1.0,1.0,(k,n,m))   
    
    return w
    
def init_z(k,n):
    '''初始化z矩阵,大小kxn ; z=w.T*x'''
    z=np.zeros((k,n) ) 
    return z
    
def init_y(k,n):
    '''初始化z矩阵,大小kxn ; z=w.T*x'''
    y=np.zeros((k,n))  
    return y
    
def init_delta(k,n):
    '''初始化delata矩阵'''
    delta=np.zeros((k,n) ) 
    return delta  

def init_delta_w(k,n,m):
    '''初始化delata矩阵'''
    delta_w=np.zeros((k,n,m))  
    return delta_w      
 
#-------------函数-----------
def get_v(my_x,my_w,my_b):
    '''局部诱导域,用于前向计算中
    my_x=np.array([1,2,0])
    my_w=np.array([[0.5,0.4,0.3],[0.5,0.6,0.2],[0.1,0.1,0.1]]) 
    y=get_v(my_x,my_w)
    y1=x1*w11+x2*w21,y2=x1*w12+x2*w22
    y1=dot(my_x,my_w[:,0])
    y=dot(x,w)=array([ 1.5,  1.6]) #第1行*第1列，第1行*第2列
    '''
    my_y=np.dot(my_x,my_w)+my_b 
    #或者： my_y=np.dot(my_w.T,my_x)+my_b #注意转置
    return np.array(my_y) 
    
def sigmod(my_x):
    y=1.0/(1+np.exp(-my_x))    
    return y

def delta_sigmod(flag,my_d,my_y,my_delta,my_w):
    ''' flag==1,表示输出层'''  
    
    deriv =my_y*(1.0-my_y) #u这个公式要求my_y的范围（0，1） 
     
    #输出层
    if flag==1:
        v_delta=deriv *(my_d-my_y) 
    #隐藏层
    else: 
        n=len(my_delta)
        array_t=np.zeros(n)
        for j in xrange(0,n):
            array_t[j]=np.dot(my_w[j],my_delta)    
        v_delta=deriv*array_t 
        
    return v_delta

def tanh(x):  
    return np.tanh(x)  
    
def delta_tanh(flag,my_d,my_y,my_delta,my_w):   
    deriv = 1.0 - np.square(my_y)
     #输出层
    if flag==1:
        v_delta=deriv *(my_d-my_y) 
    #隐藏层
    else: 
        n=len(my_delta)
        array_t=np.zeros(n)
        for j in xrange(0,n):
            array_t[j]=np.dot(my_w[j],my_delta) 
        v_delta=deriv*array_t  
     
    return v_delta     
    
def sgn(x) :
    if x > 0.0 :
        return 1
    else :
        return 0
        

        
#-------------训练-----------  
def train_forword(i_xi):
    '''一个样本的计算,前向计算
    i_xi的结构为向量，如(b,2,3)'''
    
    #除最后输出层，其他每一层x,y的结构都是一致的，即元素个数是一样的。
    #在单层感知器是dot(w,x)=w1*x1+w2*x2=y(只输出一个值)，
    #在隐藏层，则是x=(1,2) w=([0.3,0.5],[0.4,0.6]),则y1=x1*w11+x2*w21
    #即o=np.dot(y,w) ,y,w顺序不能乱
    global ann_w    
    global ann_b
    global ann_z 
    global ann_y
    global ann_delta  
    
    
    k=len(ann_w)
    #前向计算，激励传播;     
    for level in xrange(0,k): 
        if level==0 : #第一层
            z=get_v(i_xi,ann_w[0],ann_b[0]) 
            y=ann_func(z)
            ann_z[level]=z
            ann_y[level]=y 
        elif level<k-1:  #中间层
            z=get_v(ann_y[level-1],ann_w[level],ann_b[level])
            y=ann_func(z)
            ann_z[level]=z
            ann_y[level]=y  
        elif level==k-1:  #输出层
        #输出层只有一个目标，注意，不是取ann_w[level][0]，而是ann_w[level][:,0]
            z=get_v(ann_y[level-1],ann_w[level][:,0],ann_b[level][0])
            y=ann_func(z)
            #y=sgn(y)
            ann_z[level][0]=z
            ann_y[level][0]=y  
            res = y 

    return res

    
def train_back(i_xi,i_d,eta,alpha):
    '''一个样本的计算,后向计算
    i_xi的结构为向量，如(b,2,3)'''
    
    #除最后输出层，其他每一层x,y的结构都是一致的，即元素个数是一样的。
    #在单层感知器是dot(w,x)=w1*x1+w2*x2=y(只输出一个值)，
    #在隐藏层，则是x=(1,2) w=([0.3,0.5],[0.4,0.6]),则y1=x1*w11+x2*w21
    #即o=np.dot(y,w) ,y,w顺序不能乱
    global ann_w
    global ann_b
    global ann_y
    global ann_z 
    global ann_delta  
    
    
    k=len(ann_w)
    n=len(ann_delta[0]) 

    #后向计算     
    for level in xrange(k-1,-1,-1): 
        
        tmp=np.zeros_like(ann_w[0]) 
        #输出层,delta只有一个值 
        if level==k-1: 
            ann_delta[level,0]=delta_func(1,i_d,ann_y[level][0] ,None,None)    
            #print 'x=',i_xi[0:2],'y=',ann_y[level][0],'d=',i_d,'delta=',ann_delta[level],'eta=',eta  
            #计算delta_w 
            for j in xrange(0,n): 
                tmp[j]=eta * ann_delta[level] * ann_y[level-1,j] 
                #print 'tmp['+str(j)+']=',tmp[j],ann_y[level-1,j],ann_delta[level,0] 
                
            
        elif level>0: ##隐藏层
            ann_delta[level]=delta_func(3,None,ann_y[level],ann_delta[level+1],ann_w[level+1]) 
            
            for j in xrange(0,n):
                tmp[j]=eta * ann_delta[level] * ann_y[level-1,j]  
            
        else: #输入层
            ann_delta[level]=delta_func(3,None,ann_y[level],ann_delta[level+1],ann_w[level+1]) 
            
            for j in xrange(0,n):
                tmp[j]=eta * ann_delta[level] * i_xi[j] 
        
        #应用动量参数alpha         
        delta_w[level]= tmp + delta_w[level]*alpha 
        #权重更新
        #ann_w[level]=ann_w[level]+delta_w[level]
        #ann_b[level] = ann_b[level] + eta * ann_delta[level]
        
    #权重更新      
    ann_w=ann_w+delta_w
    ann_b = ann_b + eta * ann_delta
    #print 'ann_delta[',level,'] =',ann_delta[level] 
    #print 'delta_w[',level,'] =',delta_w[level] 
        
    if 0 :
        print '------------------------' 
        print 'd=',i_d
        print 'ann_y[-1]=',ann_y[-1] 
        print 'ann_delta[-1]=',ann_delta[-1] 
        print 'delta_w[-1]=',delta_w[-1]
        print 'ann_w[-1][:,0] =',ann_w[-1][:,0] 
        print 'ann_b[-1]=',ann_b[-1] 
        print '------------------------'  
        print 'ann_y[-2]=',ann_y[-2]
        print 'ann_delta[-2]=',ann_delta[-2] 
        print 'delta_w[-2]=',delta_w[-2]  
        print 'ann_b[-2]=',ann_b[-2]  
        
def init(k,n) :
    global ann_w
    global ann_b
    global ann_z
    global ann_y
    global ann_delta
    global delta_w 
    global ann_func
    global delta_func 
    
    #第一步，初始化  
    ann_w=init_w(k,n,n)
    #ann_b=init_b(k)
    ann_b=init_b(k,n)
    ann_z=init_z(k,n)
    ann_y=init_y(k,n)
    ann_delta=init_delta(k,n) 
    delta_w=init_delta_w(k,n,n) 
    
    #激励传播函数
    ann_func=tanh    
    delta_func=delta_tanh    
            

def train(my_x,my_d,eta_0,alpha_0):
    '''训练多次，多次迭代
    k代表层级，等于隐藏层+1，等于ann_w的层级
    n代表每层的节点个数'''   
    global ann_w
    global ann_b
    global ann_z
    global ann_y
    global ann_delta
    global delta_w
    
    train_count=0   
    last_mse=10000
    err=[] 
    
    
    cnt = int(len(my_x)/5)
    #第二步，
    while True: 
        #学习率
        eta=eta_0/(1+float(train_count)/r)
        #
        alpha=alpha_0         
        train_count+=1 
        
        mse =0
        #打乱样本,并抽样  
        for i in xrange(0,cnt):
            #随机选取一行，对神经网络进行更新  
            i = np.random.randint(my_x.shape[0])   
            #单样本训练
            y=train_forword(my_x[i])
            #print my_d[i] , y
            e= np.square(my_d[i]-y)*0.5             
            mse+= e
            
            if i==cnt-1:
                #mse=mse/float(i))
                err.append(mse)
                #口袋算法,保存最优的ann_w
                if mse<last_mse:
                    best_w=ann_w
                    best_b=ann_b
                    last_mse=mse
            
            #后向计算    
            train_back(my_x[i],my_d[i],eta,alpha) 
            
            
        if  train_count==1:
            print u'--开始第1次训练--##误差为%f'%(mse) 
            #print 'ann_w[0]=',ann_w[0] 
        elif train_count%50 ==0 :
            print u'--开始第%d次训练--##误差为%f'%(train_count,mse) 
            if 0 :
                print '---------------------' 
                #print 'ann_w[0]=',ann_w[0]  
                print 'ann_y=',ann_y
                #print 'ann_z=',ann_z
                #print 'ann_b=',ann_b
                #print 'ann_delta=',ann_delta 
                
        if mse<expect_e or train_count>=maxtrycount:
            print u'--开始第%d次训练--##误差为%f'%(train_count,mse) 
            #print 'best_w=',best_w 
            #print 'best_b=',best_b 
            #print 'ann_y=',ann_y 
            
            break
    return (best_w,best_b,err)
         
def sim(i_xi,best_w,best_b): 
     
    global ann_z 
    global ann_y   
    
    k=len(best_w)
    #前向计算，激励传播;     
    for level in xrange(0,k): 
        if level==0 : #第一层
            #ann_y[0]=i_xi
            z=get_v(i_xi,best_w[0],best_b[0])
            y=ann_func(z)
            ann_z[level]=z
            ann_y[level]=y 
        elif level<k-1:  #中间层
            z=get_v(ann_y[level-1],best_w[level],best_b[level])
            y=ann_func(z)
            ann_z[level]=z
            ann_y[level]=y  
        elif level==k-1:  #输出层 
            z=get_v(ann_y[level-1],best_w[level][:,0],best_b[level][0])
            y=ann_func(z)   
            res = sgn(y)
            #print y,res

    return res    
#-------------参数初始化-------------

#学习率参数
eta_0=0.1    
alpha_0=0.005
#终止条件控制
expect_e=0.1  #
maxtrycount=1000  #最大训练次数
#退火因子 
if maxtrycount>=100:
    r= round(maxtrycount/10,0)
else:
    r=10  
#------------- 测试 ----------- 
x1=np.arange(0,10,0.1)
y1=np.sin(x1)
y2=np.cos(x1)
train_x=[]
d=[]   
for i in xrange(0,len(x1)):
    train_x.append([x1[i],y1[i]])
    d.append(1)
    train_x.append([x1[i],y2[i]])
    d.append(0)
     
train_x=np.array(train_x)
d=np.array(d) 


#输入数据处理
k=4 #隐藏层+输入层
m=8  #隐藏层节点数
#input=init_input(train_x)
ann_x=init_input(train_x,[[-1,10],[-1,1]],m) 
ann_d=init_d(d,[0,1])
#print ann_d
#init w,delta
init(k,m) 

#训练  
#print ann_x[0:4]
(best_w,best_b,err)=train(ann_x,ann_d,eta_0,alpha_0)
#仿真
out=np.zeros_like(d)

for i in xrange(len(ann_x)):
    out[i]=sim(ann_x[i],best_w,best_b)   
    
#画图
my_mean=np.mean(out)  
x_max=np.max(train_x[:,0])+1  
x_min=np.min(train_x[:,0])-1  
y_max=np.max(train_x[:,1])+2  
y_min=np.min(train_x[:,1])-2  


#误差曲线 
plt.subplot(211) 
plt.plot(range(len(err)),err)  
    
#可视化图  
plt.subplot(212) 
plt.xlim(x_min,x_max)  
plt.ylim(y_min,y_max)  
for i in xrange(0,len(train_x)):  
    if out[i]>0.5:  
        plt.plot(train_x[i,0],train_x[i,1],'ro')  
    else:  
        plt.plot(train_x[i,0],train_x[i,1],'g*')  
  
plt.show() 