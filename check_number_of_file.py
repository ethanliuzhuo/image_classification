# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 09:06:59 2020

@author: ethan
"""


import os

path ='E:\\data\\classification2\\train'
# path = 'D:\\data\\EPIC_KITCHENS_2018\\frames_rgb_flow\\flow\\test'
for root, dirss, filenamess in os.walk(path):
    # print(root)
    dir_count = 0
    file_count = 0
    for roots, dirs, filenames in os.walk(root):
        for dir in dirs:
            dir_count += 1
        for file in filenames:
            file_count += 1
    print ('file_count ', file_count)