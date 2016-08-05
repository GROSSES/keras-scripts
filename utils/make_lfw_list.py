# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 15:14:08 2016

@author: Bigmoyan
"""

import os
import random

path = 'lfw'
print os.curdir
f_single = open('singles.lst','w')
f_couple = open('couples.lst','w')
# record single image and couple images separately
count = 0
for people in os.listdir(path):
    img_list = os.listdir(os.path.join(path,people))
    if len(img_list)==1:
        f_single.write(os.path.join(path,people,img_list[0])+'\n')
    else:
        for i in range(len(img_list)):
            f_single.write(os.path.join(path,people,img_list[0])+'\n')
            for j in range(i+1,len(img_list)):
                count+=1
                print "adding %dth postive samples.."%count
                f_couple.write(os.path.join(path,people,img_list[i])+' '+os.path.join(path,people,img_list[j]) + ' 1\n')

f_single.close()

f_single = open('singles.lst','r')
# make negative samples:
single_imgs = f_single.readlines()
neg_count = 0
while neg_count<count:
    i = random.randint(0,len(single_imgs)-1)
    j = random.randint(0,len(single_imgs)-1)
    if single_imgs[i].split(os.path.sep)[1] != single_imgs[j].split(os.path.sep)[1]:
        f_couple.write(single_imgs[i].strip('\n')+' '+single_imgs[j].strip('\n') + ' 0\n')
        neg_count+=1
        print "adding %dth negtive samples.."%neg_count
f_single.close()
f_couple.close() 
        
