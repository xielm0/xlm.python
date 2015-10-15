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

isdebug=False
#-------------参数初始化-------------

#学习率参数
eta_0=0.05  #初始学习率
eta=eta_0
#动量参数 momentum
alpha_0=0.005
alpha=alpha_0
#终止条件控制
expect_e=0      #
maxtrycount=1000  #最大训练次数
#退火因子 
if maxtrycount>=100:
    r=maxtrycount/10
else:
    r=5
#神经网络层数设置
#隐藏层，输入层算隐藏层第一层 
hidelevel_count=4
alllevel_count=hidelevel_count+1
#隐藏层的节点数 n, n>=x的节点数 ，
n=8  

#-------------数据规约处理-------------
#得到规约后的训练数据 x 
    
def init_input(my_input):
    global train_max
    global train_min 
    
    my_x=np.array(my_input)    
    xi_len=len(my_x[0]) #一个样本的元素的个数 
    
    train_max=[]
    train_min=[] 
    for i in xrange(0,xi_len):
        max_value=float(max(my_x[:,i]))
        min_value=float(min(my_x[:,i])) 
        train_max.append(max_value)
        train_min.append(min_value) 
    
    train_max=np.array(train_max)  #[10,340]
    train_min=np.array(train_min)  
    
    #y=(x-mean)/(max-min)
    x=copy.deepcopy(my_x)
    for i in xrange(0,len(my_x)):
        for j in xrange(0,xi_len):
            x[i][j]=(x[i][j]-train_min[j])/(train_max[j]-train_min[j])
 
    return x*2-1
    
#对输入的散乱数据，经过一个矩阵变成一个均匀分布的数据，也就是处理后的数据方差接近1。 
def init_input2(my_input):
    '''扩维，对输入数据进行处理，输入数据dot一个数值后变成神经网络的输入
    同时要求方差接近1'''
    global train_max
    global train_min 
    global first_w
    global n
    
     
    my_x=np.array(my_input)    
    xi_len=len(my_x[0]) #一个样本的元素的个数  
    x_len=len(my_input)
    
    #在这里是为了将集中的数据打散，我们测试，如果数据rand在（-1，1）,那么他的标准差在0.55-0.6之间
    #先将数据转换成（-1，1）的数据，方法：2*x/max-1 
    
    x=2*my_x/train_max-1
    #通dot一个数组来扩维
    inti_cnt=10 
    tmp_w1=np.random.uniform(-1.0,1.0,(inti_cnt,xi_len,n))
    #最小方差
    min_var=100
    y=np.zeros((x_len,n))
    for i in xrange(0,inti_cnt):
        for j in xrange(0,x_len): 
            y[j]=np.dot(x[j],tmp_w1[i])
            
        my_var=np.var(y)
        if abs(my_var-0.57)<min_var:
            choice_w=tmp_w1[i]
            min_var=abs(my_var-0.57) 
            
    first_w=choice_w
    print 'first_w=',first_w,'min_var=',min_var         
#-------------生成权值矩阵------------
#隐藏层与输入层的节点数可以不一致，先看作一致，做程序编程
#权值就是神经网络中的线，在单层感知器中，权值的个数与x[0]的个数一样
#多层感知器则是一个多维数组，如果第一层的输入为2个元素，第二层也为2个元素，那么这一层的权值就是2*2=4个
#所以，权值矩阵的shape=（层数,[n,n],len(d[o])）
#在这里，一层的权值是[],一共有 alllevel_count-1 层,但最后一层，只取第一行。
#使用随机生成
#-------------梯度初始化，输出初始化-----------
#神经网络结构：输入层、中间层、输出层(最终结果层)
#输入层、中间层*权重的结果都存放在临时的输出层中,最后输出时，再对最后的输出层进行sign处理
#梯度初始化，赋值为0，梯度矩阵的结构与y的结构一样
#输出层初始化，与权值矩阵的层数一样  
  
def init_w():
    '''随机生成n个矩阵，然后选择平均值最接近0的 '''
    global hidelevel_count
    global alllevel_count
    global n
    global ann_w
    global ann_y
    global ann_delta
    global ann_b
    global delta_w
    global delta_b
    
    min_mean=1
    init_cnt=300 
    
    ann_init_w=np.random.uniform(-1.0,1.0,(init_cnt,hidelevel_count,n,n)) 
    
    for i in xrange(0,init_cnt):
        #最后一层，只有第一列不为空，其余为空
        for j in xrange(1,n):  
            ann_init_w[i,hidelevel_count-1][:,j]=0
        
        #第一层，只有前2行不为空，其余为空
        for j in xrange(2,n):  
            ann_init_w[i,0][j,:]=0
        #均值    
        v_mean=abs(np.mean(ann_init_w[i]))
        if v_mean<min_mean:
            min_mean=v_mean
            choice_i=i
        if abs(np.mean(ann_init_w[i]))<0.008:break    
   
    ann_w=ann_init_w[choice_i]
    #其他的
    ann_b=np.zeros((alllevel_count,n))
    ann_y=np.zeros((alllevel_count+1,n))
    ann_delta=np.zeros((alllevel_count,n)) 
    delta_w=np.zeros_like(ann_w)
    delta_b=np.zeros_like(ann_b)
    
#ann_w=np.random.uniform(-1.0,1.0,(hidelevel_count,n,n)) 
#ann_w=np.random.uniform(-0.5,0.5,(5,2,2)) 
#第一层，只有前2行不为空，其余为空
#最后一层，只有第一列不为空，其余为空
#for j in xrange(1,n):  
#    ann_w[hidelevel_count-1][:,j]=0
#for j in xrange(2,n):  
#    ann_w[0][j,:]=0
#    
#ann_delta 

#print ann_y
#-------------函数-----------
def get_v(my_x,my_w):
    '''局部诱导域,用于前向计算中
    my_x=np.array([1,2])
    my_w=np.array([[0.5,0.4],[0.5,0.6]]) 
    y=get_v(my_x,my_w)
    y1=x1*w11+x2*w21,y2=x1*w12+x2*w22
    y1=dot(x,w[:,1])
    y=dot(x,w)=array([ 1.5,  1.6]) #第1行*第1列，第1行*第2列
    '''
#    my_y=[]
#    for j in xrange(n): #n,隐藏层的节点数
#        my_y.append([])
#        my_y[j]=np.dot(my_x,my_w[:,j])
    my_y=np.dot(my_x,my_w)
    return np.array(my_y) 
    
def ann_sigmod(my_x):
    a=6.0
    y=1.0/(1+np.exp(-a*my_x))
    
    return y

def ann_delta_sigmod(flag,my_y,my_d,my_delta,my_w):
    global n
    a=6.0
    v_a=a*my_y*(1.0-my_y) #u这个公式要求my_y的范围（0，1）
    #print 'my_y=',my_y
    #输出层
    if flag==1:
        v_delta=v_a *(float(my_d)-my_y)
    #输出层的上一层，即隐藏层的最后一层    
    elif flag==2:
        array_b=my_delta[0]*my_w[:,0] 
        v_delta=v_a*array_b
        
    #隐藏层
    else: 
        #神经网络是篱笆网络
        #delta11=w11*delta21+w12*delta22,delta12=w21*delta21+w22*delta22, delta1=[delta11,delta12 ]
        #w=[[1,2],[3,4]],delta=[2,3],np.dot(w,delta)=array([ 8, 18]),np.dot(w[0],delta)=8    
        array_b=[]
        #n=len(my_delta) #n,隐藏层的节点数=len(my_delta) 
        for j in xrange(0,n): 
            array_b.append([]) 
            array_b[j]=np.dot(my_w[j],my_delta) 
        array_b=np.array(array_b)    
        v_delta=v_a*array_b 
        
    return v_delta
    
def ann_atanh(my_x):
    '''激活函数为双曲正切函数
    输入参数： my_x ,向量 
    '''
    atanh_a=2 #>0
    atanh_b=1 #>0
    my_v=atanh_a*np.tanh(atanh_b*my_x)
    return my_v
        
def ann_delta_atanh(flag,my_y,my_d,my_delta,my_w):
    '''局部梯度delta计算
    if flag=1:
    my_y表示本层y,my_delta表示后一层的delta,my_w表示后一层的w
    example:my_delta=[1,0],my_w=[[0.3,0],[0.5,0]]
    return:1*[0.3,0.5]
    example:my_delta=[0.3,0.4] ,my_w=[[0.3,0.4],[0.5,0.6]]
    retrun:tmp=[tmp1,tmp2],tmp1=w11*delta1+w12*delta2,tmp2=w21*delta1+w22*delta2,  
    '''
    global n
    
    #atanh_a>0
#    atanh_a=1.7159  
#    atanh_b=2/3.0
    
    atanh_a=2  
    atanh_b=1
    
    v_a=(float(atanh_b)/atanh_a)*(atanh_a-my_y)*(atanh_a+my_y)
    #输出层 
    if flag==1:
        v_delta=v_a*(my_d-my_y)
    #输出层的上一层，即隐藏层的最后一层    
    elif flag==2:
        array_b=my_delta[0]*my_w[:,0] 
        v_delta=v_a*array_b
        
    #隐藏层
    else: 
        #神经网络是篱笆网络
        #delta11=w11*delta21+w12*delta22,delta12=w21*delta21+w22*delta22, delta1=[delta11,delta12 ]
        #w=[[1,2],[3,4]],delta=[2,3],np.dot(w,delta)=array([ 8, 18]),np.dot(w[0],delta)=8    
        array_b=[]
        #n=len(my_delta) #n,隐藏层的节点数=len(my_delta) 
        for j in xrange(0,n): 
            array_b.append([]) 
            array_b[j]=np.dot(my_w[j],my_delta) 
        
        array_b=np.array(array_b)
        v_delta=v_a*array_b 
        
    return v_delta

#激励传播函数
sigmoid_func=ann_sigmod    
delta_func=ann_delta_sigmod   
    
def output_func(my_y):
    '''
    对最终输出进行最后处理
    my_y=[0.5,0,0]
    result=[1,0,0],与my_y的结构一样
    如果是logstic函数，因为输出的值是在(0,1),如果是 tanh 函数，则范围是(-1,1)
    ''' 
    global d_mean
    
    if my_y>=0.5:
        rs=1
    else:
        rs=0
    return rs;             
        
#-------------训练-----------  
def train_forword(i_xi,i_d):
    '''一个样本的计算,前向计算
    i_xi的结构为向量，如(2,3)'''
    
    #除最后输出层，其他每一层x,y的结构都是一致的，即元素个数是一样的。
    #在单层感知器是dot(w,x)=w1*x1+w2*x2=y(只输出一个值)，
    #在隐藏层，则是x=(1,2) w=([0.3,0.5],[0.4,0.6]),则y1=x1*w11+x2*w21
    #即o=np.dot(y,w) ,y,w顺序不能乱
    global ann_w
    global ann_y
    global ann_b
    global ann_delta
    global delta_w 
    global delta_b
    global eta 
    global alpha
    global n
    
    xi_len=len(i_xi) 
    #前向计算，激励传播;     
    #注意，在这里将输出层分成2层，一层存放wx+b的和，一层存放经过0，1处理的最终结果。让ann_y增加一层
    for level in xrange(0,alllevel_count+1):
        if level==0 : #第一层
            #ann_y[0]=i_xi
            for j in xrange(n):
                if j<=xi_len-1:
                    ann_y[0][j]=i_xi[j]
                else:  
                    ann_y[0][j]=0.0
        elif level==alllevel_count:  #输出层2 ,对y进行sign处理 
            y=output_func(ann_y[level-1][0]) 
            ann_y[level,0]=y
        elif level==alllevel_count-1:  #输出层1，
            y=sigmoid_func(get_v(ann_y[level-1],ann_w[level-1][:,0]) )
#换成线性激活函数          
#            y=sigmoid_func(np.sum(ann_y[level-1]*ann_w[level-1][:,0])+ann_b[level-1][0] )
            ann_y[level,0]=y 
        else: #中间层,返回的还是一个向量
            y=sigmoid_func(get_v(ann_y[level-1],ann_w[level-1]))
#            y=sigmoid_func(get_v(ann_y[level-1],ann_w[level-1])+ann_b[level-1])
            ann_y[level]=y 
        #print y
    #误差 d-y,如果d-y=0,代表权重不需要变化 
    v_o=ann_y[alllevel_count-1][0]  
    v_o_end=ann_y[alllevel_count,0]
    my_e=i_d-v_o_end  
        
    #返回误差   
    return my_e 
    
def train_back(i_xi,i_d):
    '''一个样本的计算,后向计算
    i_xi的结构为向量，如(2,3)'''
    
    #除最后输出层，其他每一层x,y的结构都是一致的，即元素个数是一样的。
    #在单层感知器是dot(w,x)=w1*x1+w2*x2=y(只输出一个值)，
    #在隐藏层，则是x=(1,2) w=([0.3,0.5],[0.4,0.6]),则y1=x1*w11+x2*w21
    #即o=np.dot(y,w) ,y,w顺序不能乱
    global ann_w
    global ann_y
    global ann_b
    global ann_delta
    global delta_w 
    global delta_b
    global eta 
    global alpha
    global n
     

    #后向计算
    #delta_3=f(y4,delta4,w3),之前犯了个错误=(delete4,w4)
    #xrange(hidelevel_count-1,-1,-1)而不是xrange(hidelevel_count-1,0,-1)
    for level in xrange(alllevel_count-1,0,-1): #delta第一层不需要计算
        #输出层,delta只有一个值
        if level==alllevel_count-1: 
            tmp_delta=delta_func(1,ann_y[level][0] ,i_d,None,None)
            ann_delta[level,0]=tmp_delta  
#            print 'y=',v_o,'d=',i_d,'delta=',tmp_delta
        elif level==alllevel_count-2: 
            #隐藏层最后一层，delta=1个数字*1个权重（一维数组）=1个数组
            tmp_delta=delta_func(2,ann_y[level],i_d,ann_delta[level+1],ann_w[level]) 
            ann_delta[level]=tmp_delta 
        else: #隐藏层
            tmp_delta=delta_func(3,ann_y[level],i_d,ann_delta[level+1],ann_w[level])
            ann_delta[level]=tmp_delta
            
        
    #权重更新
    #w3=old_w3+eta*delta4*y3 ,eta学习率，难点在：delta4*y3如何计算
    #在最后一层时，若delta5=2,y4=[0.4,0.5];w4=[[0.3,0],[0.4,0]],
    #则delta4*y3返回一个矩阵，且第2列=0.那么delta5=[2,0],delta5*y4=0.4*[2,0],0.5*[2,0]
    #以此类推。
    #b值更新  newb_b=b+eta*delta
    #在最后一层时，b=1,delta=2,那么new_b=b+delta=1+2=3
    #隐藏层与输出层，delta=[2,3],b=[1,4] ,new_b=b+delta   
    old_delta_w=copy.deepcopy(delta_w)
    old_delta_b=copy.deepcopy(delta_b)
    tmp=np.zeros_like(ann_w[0]) #delta4*y3的临时变量
    for level in xrange(hidelevel_count-1,-1,-1):
        #deta_w=eta*delta4*y3 
        for j in xrange(0,n):
            tmp[j]=ann_y[level,j]*ann_delta[level+1] 
        
        #应用动量参数alpha
        #delta<0,y属于(0,1),那么delta_w变的更少了。
        delta_w[level]=(1-alpha)*eta*tmp+old_delta_w[level]*alpha       
        ann_w[level]=ann_w[level]+delta_w[level]
        #b
        delta_b[level]=(1-alpha)*eta*ann_delta[level+1]+old_delta_b[level]*alpha
        ann_b[level]=ann_b[level]+delta_b[level]
        
    if 0 :
        print '---------------------'
        print 'x=',i_xi,'d=',i_d
        print 'ann_y=',ann_y
        print 'ann_delta=',ann_delta
        print 'delta_w=',ann_b 
        print 'ann_w[last]=',ann_w[hidelevel_count-1] 
        #print 'delta_b=',delta_b   
        

def train(my_x,my_d):
    '''训练多次，多次迭代'''
    global eta  
    global alpha  
    global best_w
    
    train_count=0   
    x_len=len(my_x)
    last_mse=10000
    err=[]
    while True: 
        #学习率
        eta=eta_0/(1+float(train_count)/r)
        #动量参数，动量参数先低后高
        alpha=alpha_0*(1+float(train_count)/r)
        
        train_count+=1 
        #如果有需要，可以在这打乱样本
       
        #train 
        mse=0 
        for i in xrange(0,x_len):
            #先做处理
#            new_x=np.dot(my_x[i],first_w)
            #单样本训练
            e=train_forword(my_x[i],my_d[i])             
            mse+=np.power(e,2)
            if i==x_len-1:
                #mse=np.sqrt(mse/float(i))
                err.append(mse)
                #口袋算法,保存最优的ann_w
                if mse<last_mse:
                    best_w=ann_w
                    last_mse=mse
            #后向计算    
            train_back(my_x[i],my_d[i]) 
            i+=1
            
        if  train_count==1:
            print u'--开始第1次训练--##误差为%f'%(mse)   
        elif train_count%50 ==0 :
            print u'--开始第%d次训练--##误差为%f'%(train_count,mse) 
            #print 'delta_w=',delta_w 
#            print 'delta_b=',delta_b  
        if mse<expect_e or train_count>=maxtrycount:
            print u'--开始第%d次训练--##误差为%f'%(train_count,mse) 
            #print 'ann_w=',ann_w   
            #print 'ann_b=',ann_b  
            
            break
    return err
         
#-------------仿真-----------
def sim(my_x):
    global best_w 
    global n
    global alllevel_count
    global train_max
    global train_min
    
    #输出矩阵
    ann_y=np.zeros((alllevel_count,n))    
    #数据规约
    x=np.array(my_x)    
    for i in xrange(len(x)): 
        tmp=(x[i]-train_min[i])/(train_max[i]-train_min[i])
        if tmp>1.0:
            x[i]=1
        elif tmp<0.0:
            x[i]=0
        else:
            x[i]=tmp 
    #前向计算
    for level in xrange(alllevel_count+1):
        if level==0 : #第一层
            #ann_y[0]=i_xi
            for j in xrange(n):
                if j<=len(my_x)-1:
                    ann_y[0][j]=my_x[j]
                else:  
                    ann_y[0][j]=0.0 
        elif level==alllevel_count:  #输出层2 ,输出层采用线性激活函数，不需要s型函数 
            y=output_func(ann_y[level-1][0]) 
            result=y
        elif level==alllevel_count-1:  #输出层1
            y=sigmoid_func(get_v(ann_y[level-1],best_w[level-1][:,0]) )
## 采用线性激活函数，不需要s型函数             
#            y=sigmoid_func(np.sum(ann_y[level-1]*best_w[level-1][:,0])+ann_b[level-1][0] )
            ann_y[level,0]=y  
        else: #中间层,返回的还是一个向量 
            y=sigmoid_func(get_v(ann_y[level-1],best_w[level-1]) )
#            y=sigmoid_func(get_v(ann_y[level-1],best_w[level-1])+ann_delta[level-1] )
            ann_y[level]=y 
            
    return result 
#------------- 测试 ----------- 
x1=np.arange(0,10,0.1)
y1=sin(x1)
y2=cos(x1)
train_x=[]
d=[]   
for i in xrange(0,len(x1)):
    train_x.append([x1[i],y1[i]])
    d.append(1)
    train_x.append([x1[i],y2[i]])
    d.append(0)
     
train_x=np.array(train_x)
target=np.array(d) 


#输入数据处理
input=init_input(train_x)
#权重矩阵等初始化
init_w()
print 'mean_w=',np.mean(ann_w),
print 'mean_w0=',np.mean(ann_w[0])
#训练  
err=train(input,target)
#仿真
out=np.zeros_like(target)
for i in xrange(len(input)):
    out[i]=sim(input[i]) 
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
for i in xrange(0,len(input)):  
    if out[i]>0.5:  
        plt.plot(train_x[i,0],train_x[i,1],'ro')  
    else:  
        plt.plot(train_x[i,0],train_x[i,1],'g*')  
  
plt.show() 