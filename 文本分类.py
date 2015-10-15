#!/usr/bin/env python
#-*- coding: utf-8 -*-
#code:myhaspl@qq.com
#10-5.py
#bayes文本分类
#本程序仅做机器学习研究
#本程序对新闻爬取的工作原理与搜索引擎相同，通过分析链接
#直接搜索新闻,计算词条概率


import numpy as np
import jieba
import urllib2
from bs4 import BeautifulSoup
import re

#读取网上新闻搜索目录
txt_class=[]
myclassfl = open('ClassList.txt')  
try:  
    myclass_str = myclassfl.read() 
    myclass_str=unicode(myclass_str,'gbk')
    myclass_text=myclass_str.split()
    for ii in xrange(0,len(myclass_text),2):
        print ".",
        txt_class.append((myclass_text[ii],myclass_text[ii+1]))            
finally:  
    myclassfl.close()

links=[]
#分类别爬取网页，生成词条数据
for ci in xrange(0,len(txt_class)):
    print u"\n爬取%s类网页:%s" % (txt_class[ci][0],txt_class[ci][1])
    links.append([])    
    pattern = re.compile(r'(.*?)/\d+\.shtml')
    purl=txt_class[ci][1]
    page=urllib2.urlopen(purl)
    soup = BeautifulSoup(page)
    for link in soup.find_all('a'): 
        mylink=link.get('href')
        match = pattern.match(mylink)
        if match and mylink.find("hd")<0:
            basestr="http://www.chinanews.com"
            if mylink.find("chinanews.com")<0:
                mylink=basestr+mylink
                print  mylink
                links[ci].append(mylink)

#提取正文内容
ybtxt=[]
print u"\n提取正文内容"
for ci in xrange(0,len(txt_class)):
    ybtxt.append([])
    print ".",
    for mypage in links[ci]:
        try:
            my_page=urllib2.urlopen(mypage)
        except: 
           continue
            
        
        my_soup = BeautifulSoup(my_page,from_encoding="gb2312")
        my_tt=my_soup.get_text("|", strip=True)
        my_txt=my_tt
        my_fs=u'正文|'
        my_fe1=u'【编辑'
        my_fe2=u'标签'
        zw_start=my_txt.find(my_fs)+8
        last_txt=my_txt[zw_start:len(my_txt)]
        zw_end=last_txt.find(my_fe1)
        if zw_end<0:
            zw_end=last_txt.find(my_fe2)
        page_content=my_txt[zw_start:zw_start+zw_end]
        page_content=page_content.replace(r'_acK({aid:1807,format:0,mode:1,gid:1,serverbaseurl:"me.afp.chinanews.com/"});','').replace('|','').replace(r'{aid:1805,format:0,mode:1,gid:1,serverbaseurl:"me.afp.chinanews.com/"}','').replace('cK();','')
        page_content=page_content.replace(u'1807：新闻通发页 大画','').replace(u'标签：','').replace(u'评论','').replace(u'正文start编辑姓名start编辑姓名','').replace(u'正文start','')
        if len(page_content.strip())>0:
            try:
                print my_soup.title.string.encode('gb2312')
                page_content=my_soup.title.string+page_content
            except:
                print "...."
            finally:
                print "-done."
                ybtxt[ci].append(page_content)

#分析正文内容
print u"\n分析正文内容..."

#停用词字典
f_stop = open('stopwords.txt')  
try:  
    f_stop_text = f_stop.read( )
    f_stop_text=unicode(f_stop_text,'utf-8')
finally:  
    f_stop.close( ) 
f_stop_seg_list=f_stop_text.split('\n')

#分类提取正文词条
print u"\n提取正文词条..."
yb_txt=[]
for ci in xrange(0,len(ybtxt)):
    yb_txt.append([])
    for cj in xrange(0,len(ybtxt[ci])):
        yb_txt[ci].append([])
        my_str = ybtxt[ci][cj]
        my_txt=jieba.cut(my_str)            
        for myword in my_txt:
            if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>1:
                yb_txt[ci][cj].append(myword) 
        print ".",    

#词条在每个样本中出现的次数
basegl=1e-10
wordybcount={}
lbcount=np.zeros(len(yb_txt))
#整理计算词条出现次数
for i in xrange(0,len(yb_txt)):
    for j in xrange(0,len(yb_txt[i])):       
        for k in xrange(0,len(yb_txt[i][j])):
            my_word=yb_txt[i][j][k].encode('gbk')
            wordybcount.setdefault(my_word,np.repeat(0,len(yb_txt)).tolist())           
            wordybcount[my_word][i]+=1
            lbcount[i]+=1
 
#计算词条先验概率
print u"\n计算词条概率"
ybgl={}

for my_word in wordybcount.keys():
    ybgl.setdefault(my_word,np.repeat(0.,len(yb_txt)).tolist())
    for ybii in xrange(0,len(yb_txt)):
        ybgl[my_word][ybii]=basegl+wordybcount[my_word][ybii]/float(lbcount[ybii])
        print '.',
        
##读取待分类文本
print u"\n读取待分类文本"
ftestlinks=[]
ftestlinks.append(r'http://www.chinanews.com/edu/2013/09-17/5296319.shtml')          
ftestlinks.append(r'http://finance.chinanews.com/auto/2013/09-16/5290491.shtml') 
for mypage in ftestlinks:
    my_page=urllib2.urlopen(mypage)
    my_soup = BeautifulSoup(my_page,from_encoding="gb2312")
    my_tt=my_soup.get_text("|", strip=True)
    my_txt=my_tt
    my_fs=u'正文|'
    my_fe1=u'【编辑'
    my_fe2=u'标签'
    zw_start=my_txt.find(my_fs)+8
    last_txt=my_txt[zw_start:len(my_txt)]
    zw_end=last_txt.find(my_fe1)
    if zw_end<0:
        zw_end=last_txt.find(my_fe2)        
    page_content=my_txt[zw_start:zw_start+zw_end]
    page_content=page_content.replace(r'_acK({aid:1807,format:0,mode:1,gid:1,serverbaseurl:"me.afp.chinanews.com/"});','').replace('|','').replace(r'{aid:1805,format:0,mode:1,gid:1,serverbaseurl:"me.afp.chinanews.com/"}','').replace('cK();','')
    page_content=page_content.replace(u'1807：新闻通发页 大画','').replace(u'标签：','').replace(u'评论','').replace(u'正文start编辑姓名start编辑姓名','').replace(u'正文start','')
    page_content=my_soup.title.string+page_content
    print u"%s读取成功."%mypage
  
    #计算待分类文本后验概率
    print u"计算待分类文本后验概率"
    testgl=None
    wordgl=None     
    testgl=np.repeat(1.,len(yb_txt))
    if len(page_content.strip())>0:
        ftest_seg_list = jieba.cut(page_content)
        for  myword in ftest_seg_list:
            myword=myword.encode('gbk')
            if not(myword.strip() in f_stop_seg_list) and len(myword.strip())>2:
                for i in xrange(0,len(yb_txt)):                    
                    wordgl=ybgl.get(myword)
                    if wordgl:
                        if wordgl[i]<>0:
                            testgl[i]*=wordgl[i]
                            if np.min(testgl)<1e-100:
                                testgl*=1e30
                            if np.max(testgl)>1e100:
                                testgl/=float(1e30)

        #计算最大归属概率
        maxgl=0.
        mychoice=0
        for ti in xrange(0,len(yb_txt)):
            if testgl[ti]>maxgl:
                maxgl=testgl[ti]
                mychoice=ti
        print "\n\n%s\n:%s"%(mypage,txt_class[mychoice][0])