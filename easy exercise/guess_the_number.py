#_*_encoding:gb2312_*_
#-------------------------------------------------------------------------------
# Name:        guess the number
# Purpose:     随机产生一个0-9的数字，你有3次机会来猜，猜对弹出：you are very good!
#
# Author:      xlm
#
# Created:     30/07/2014
# Copyright:   (c) xlm 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from random import randint,choice

#随机产生一个0-9的数字
target=choice(range(10))
target=6
n=0
#
a=int(raw_input('请猜出我想要的数字,范围0-9：'))

while a<>target:
    n=n+1
    if n>=3:
        print '你没有机会了!'
        break

    if a>target:
        a=int(raw_input( '这个数字大了！请重新输入'))
    else:
        a=int(raw_input( '这个数字小了！请重新输入'))


else:
    print '你太NB了!'


