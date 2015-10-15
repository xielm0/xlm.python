# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:45:22 2014

@author: xlm
"""
import jieba  

#读取样本
f1 = open('war1.txt')  
try:  
    f1_text = f1.read( ) 
    f1_text=unicode(f1_text,'utf-8')
finally:  
    f1.close( ) 
f1_seg_list = jieba.cut(f1_text)     
 
#读取停用词点文本，
f_stop = open('stopwords.txt')  
try:  
    f_stop_text = f_stop.read( )
    f_stop_text=unicode(f_stop_text,'utf-8')
finally:  
    f_stop.close( ) 
f_stop_seg_list=f_stop_text.split('\n')

#去除停用词，同时构造样本词的字典，计算出词频 
all_words={}
for  myword in f1_seg_list: 
    if not(myword.strip() in f_stop_seg_list): 
        all_words.setdefault(myword,0)
        all_words[myword]+=1
#打印
print all_words  #all_words是一个字典，不能打印内容，显示的是内存编码
for word in all_words :
    print word,':',all_words[word]
    
 