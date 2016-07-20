# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 15:14:08 2016

@author: Bigmoyan

This script generate .lst list for LFW dataset
Run this scripts with LFW dataset in the work directory
Two .lst files will be generated, you can delete singles.lst
In couples.lst, each line is a couple of faces, the integer at the end of each line indicate if these two faces belongs to a single man, 1 for same and 0 for different
Note that the generated list is not shuffled, you need to randomly shuffle it before using it as training set.
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
        f_single.write(os.path.join(path,people,img_list[0])+'\n')
        for i in range(len(img_list)):
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
    if i!=j:
        f_couple.write(single_imgs[i]+' '+single_imgs[j] + ' 1\n')
        neg_count+=1
        print "adding %dth negtive samples"%neg_count
f_single.close()
f_couple.close() 
        
