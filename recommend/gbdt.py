# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 14:42:47 2016
gbdt + lR

@author: xieliming
"""

from sklearn import metrics  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor    
import numpy as np  
import random

    
#无放回随机抽样
def RandomSampling(dataMat,number):
  try:
     slice = dataMat[random.sample(xrange(0,dataMat.shape[0]-1),number),:] 
     return slice
  except:
     print 'error in RandomSampling '    
     
#calc AUC
     
def aucfunc(fact,pred):
    auc= metrics.roc_auc_score(fact,pred) 
    return auc  
    
def rmse(fact,pred):
    total_err=0
    for i in range(pred.shape[0]): 
        err=(pred[i]-train_y[i])**2
        total_err+=err*err
    return total_err/pred.shape[0]
 
 

if __name__ == "__main__":
    
    org_data1 = np.loadtxt("E:\\tmp_downloads\mixRank1.txt",dtype=np.float32) #genfromtxt
    org_data0 = np.loadtxt("E:\\tmp_downloads\mixRank0.txt",dtype=np.float32)
    org_data1_s = RandomSampling(org_data1,100000) 
    org_data0_s = RandomSampling(org_data0,100000)  
    org_data_s = np.append(org_data1_s,org_data0_s,axis=0)
    
    #数据处理
    train_y =org_data_s[:,0] 
    train_x =org_data_s[:,5:8]
    
    #train
    model = GradientBoostingClassifier(
      loss='deviance'
    , learning_rate=0.1
    , n_estimators=100
    , subsample=1
    , min_samples_split=2
    , min_samples_leaf=1
    , max_depth=3
    , init=None
    , random_state=None
    , max_features=None 
    , verbose=0 
    )  
    
    model.fit(train_x, train_y)  
   
    feature_importances = model.feature_importances_
    print feature_importances

    pred=model.predict(train_x)
   
    #new feature 
    new_train_data = train_x[:, feature_importances>0]
    
    print rmse(train_y,pred)
    print aucfunc(train_y,pred)


 


