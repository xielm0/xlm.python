# -*- coding: utf-8 -*-

from matplotlib.pyplot import *
from collections import defaultdict
import random

#function to calculate distance(点与点之间的距离)
def dist(p1, p2):
  return ((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)**(0.5)

#cluster与cluster的距离

#randomly generate around 100 cartesian coordinates
all_points=[]

#生成非重复的点
for i in range(100):
  randCoord = [random.randint(1,50), random.randint(1,50)]
  if not randCoord in all_points:
    all_points.append(randCoord)


#take radius = 8 and min. points = 8
E = 5
minPts = 5

#find out the core points
#半径内个数大于minPts则为核心点
other_points =[]
core_points=[]
plotted_points=[]
for point in all_points:
  point.append(0) # assign initial level 0
  total = 0
  for otherPoint in all_points:
    distance = dist(otherPoint,point)
    if distance<=E:
      total+=1

  if total > minPts:
    core_points.append(point)
    plotted_points.append(point)
  else:
    other_points.append(point)

#find border points
border_points=[]
for core in core_points:
  for other in other_points:
    if dist(core,other)<=E:
      border_points.append(other)
      plotted_points.append(other) #簇扩展


#implement the algorithm
cluster_label=0

for point in core_points:
  if point[2]==0:
    cluster_label+=1
    point[2]=cluster_label

  for point2 in plotted_points:
    distance = dist(point2,point)
    if point2[2] ==0 and distance<=E:
      #print point, point2
      point2[2] =point[2]


#按cluster组进行归类
cluster_list = defaultdict(lambda: [[],[]])
for point in plotted_points:
  cluster_list[point[2]][0].append(point[0])
  cluster_list[point[2]][1].append(point[1])

markers = ['+','*','.','d','^','v','>','<','p']

#plotting the clusters
i=0
print cluster_list
for value in cluster_list:
  cluster= cluster_list[value]
  i = value%10
  plot(cluster[0], cluster[1],markers[i])


#plot the noise points as well
noise_points=[]
for point in all_points:
  if not point in core_points and not point in border_points:
    noise_points.append(point)
noisex=[]
noisey=[]
for point in noise_points:
  noisex.append(point[0])
  noisey.append(point[1])
plot(noisex, noisey, "x")

title(str(len(cluster_list))+" clusters created with E ="+str(E)+" Min Points="+str(minPts)+" total points="+str(len(all_points))+" noise Points = "+ str(len(noise_points)))
axis((0,60,0,60))
show()