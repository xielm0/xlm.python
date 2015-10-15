#-------------------------------------------------------------------------------
# Name:        urlretrieve
# Purpose:
#
# Author:      xlm
#
# Created:     24/02/2014
# Copyright:   (c) xlm 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import urllib
def callbackfunc(blocknum, blocksize, totalsize):
    '''回调函数
    @blocknum: 已经下载的数据块
    @blocksize: 数据块的大小
    @totalsize: 远程文件的大小
    '''
    percent = 100.0 * blocknum * blocksize / totalsize
    if percent > 100:
        percent = 100
    print "%.2f%%"% percent

url = 'http://www.sina.com.cn'
local = 'd:\\sina.html'
res=urllib.urlretrieve(url, local, callbackfunc)
print res  #返回值打印

