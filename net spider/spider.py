# -*- coding:gb2312 -*-
import re
import urllib

#打开源代码
def get_html(url):
    try:
        page=urllib.urlopen(url)
        html=page.read()
        page.close()
        return html
    except:
        return None
#下载html
def dowload_html(url,filename):
    try:
        #urllib.urlretrieve(url,'%s.html' %(filename))
        html=get_html(url)
        f=open(filename,'w')
        if not str.strp(html):
            pass
        else:
            f.write(html)
    except:
        f.close()



#通过html内容获取url列表,pattern 为正则表达式
def getUrls(html,pattern):
    try:
        reg=re.compile(pattern)
        urls=re.findall(reg,html)
        return urls
    except:
        print 'getUrls fail'
        return []

#通过url获取html,再通过html获取url列表,pattern 为正则表达式
def getUrls_2(url,pattern):
    try:
        html=get_html(url)
        reg=re.compile(pattern)
        urls=re.findall(reg,html)
        return urls
    except:
        print 'getUrls fail'
        return []

#广度优先
def spider(start_url,pattern):
##    try:
        i=0
        urls=[] #url列表 ， 
        #用hash表更好，即 {},在后面判断是否存在时，用 urls.has_key()则效率更高。
        #因为list走的是遍历，而字典是hash
        v_url=start_url
        urls.append(v_url)
        #将url放入一个列表
        while i<=200 and i<len(urls):
            v_url=urls[i]
            if v_url:
                dowload_html(v_url,'E:\\temp\%s.html'%(str(i))) #下载html
                print str(i)+'_'+v_url
                #解析网页获取url
                urllist=getUrls_2(v_url,pattern)
                for cr_url in urllist:
                    if urls.count(cr_url)==0:   #如果列表已经存在该地址，则不插入，否则插入
                        urls.append(cr_url)
            i+=1
            print len(urls)

##    except:
##        print urls.index(v_url)




start_url='http://blog.sina.com.cn/u/2130809547'
reg=r'href="(http://blog\.sina\.com\.cn/s/blog.+?\.html)"'
spider(start_url,reg)

