# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:25:53 2015

@author: xlm
"""

import re

f=open("log.1","r")
ip_count={}
lines = f.readlines()
for line in lines:
    #ipaddress=re.compile(r'^#(((2[0-4]\d|25[0-5]|[01]?\d\d?)\.){3}(2[0-4]\d|25[0-5]|[01]?\d\d?))')
    ipaddress=re.compile(r'^#((\d+\.){3}\d+)')     
    match=ipaddress.match(line)
    if match:
        ip = match.group(1)  
        # group(1)代表返回group1的内容，即第一个()的内容，即（((\d+\.){3}\d+)）
        #如果是group(),则返回结果是：#111.172.249.84
        if(ip_count.has_key(ip)):
            ip_count[ip]+=1
        else:
            ip_count.setdefault(ip,1)
        
f.close()
for key in ip_count:
    print key+"->"+str(ip_count[key])