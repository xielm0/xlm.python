# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:24:57 2016

@author: xieliming
"""

def LFM(user_items, F, N, alpha, lambda):  
    #初始化P,Q矩阵  
    [P, Q] = InitModel(user_items, F)  
    #开始迭代  
    For step in range(0, N):  
        #从数据集中依次取出user以及该user喜欢的iterms集  
        for user, items in user_item.iterms():  
            #随机抽样，为user抽取与items数量相当的负样本，并将正负样本合并，用于优化计算  
            samples = RandSelectNegativeSamples(items)  
            #依次获取item和user对该item的兴趣度  
            for item, rui in samples.items():  
                #根据当前参数计算误差  
                eui = eui - Predict(user, item)  
                #优化参数  
                for f in range(0, F):  
                    P[user][f] += alpha * (eui * Q[f][item] - lambda * P[user][f])  
                    Q[f][item] += alpha * (eui * P[user][f] - lambda * Q[f][item])  
        #每次迭代完后，都要降低学习速率。一开始的时候由于离最优值相差甚远，因此快速下降；  
        #当优化到一定程度后，就需要放慢学习速率，慢慢的接近最优值。  
        alpha *= 0.9  