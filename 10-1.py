#!/usr/bin/env python
#-*- coding: utf-8 -*-
#code:myhaspl@qq.com
#10-1.py
import jieba

a1 = jieba.cut("我来到北京清华大学", cut_all=False) # 默认模式
print "Default Mode:", "/ ".join(a1) 

a2 = jieba.cut("我来到北京清华大学", cut_all=True)  # 全模式
print "Full Mode:", "/ ".join(a2) 

# 搜索引擎模式
a3 = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造") 
print '.'.join(a3)
print "serach Mode:",", ".join(a3)