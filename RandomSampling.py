# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 15:29:19 2016

@author: xieliming
"""

import random

def loadDataSet(fileName):	  #general function to parse tab -delimited floats
  dataMat = []				#assume last column is target value
  fr = open(fileName)
  for line in fr.readlines():
    curLine = line.strip().split('\t')
#		 fltLine = map(float,curLine) #map all elements to float()
    dataMat.append(curLine)
  return dataMat

#无放回随机抽样
def RandomSampling(dataMat,number):
  try:
     slice = random.sample(dataMat, number)	
     return slice
  except:
     print 'sample larger than population'

#有放回随机抽样
def RepetitionRandomSampling(dataMat,number):	
  sample=[]
  for i in range(number):
     sample.append(dataMat[random.randint(0,len(dataMat)-1)])
  return sample

#系统抽样  
def SystematicSampling(dataMat,number):	
  
     length=len(dataMat)
     k=length/number
     sample=[]	 
     i=0
     if k>0 :	   
     while len(sample)!=number:
      sample.append(dataMat[0+i*k])
      i+=1			
     return sample
     else :
     return RandomSampling(dataMat,number)   