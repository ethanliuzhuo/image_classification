# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:24:15 2019

@author: ethan
"""

#%%
import os
from PIL import Image
import random


## python里面+和，的区别是什么来? ----+不带空格，自带空格。
def split_dataset(resultDir, test_path, train_path):
    for dirname in os.listdir(resultDir): ## 获取当前路径下的所有文件夹及文件名。 dirname是0、1、2.。。。文件夹的名字。
        FileList = [] ## 存放图片路径的列表。
        ## 提取出前三分之一的图片。
        for dirname_picture in os.listdir(os.path.join(resultDir, dirname)): ## 获取0、1、2.。。文件夹下的文件名。
            FileList.append(os.path.join(resultDir, dirname, dirname_picture))
            
        random.shuffle(FileList)
    ## 将前十分之一的图片移动到test-images文件夹下的对应文件夹下。
        for dirpath_picture in FileList[:len(FileList)//10]: ## dirpath_picture是图片的相对路径.\\result\\0\\picture.png。注意要用'//'进行整出。float不支持。
            img = Image.open(dirpath_picture)
        ## 判断文件夹是否已经创建。
            path = os.path.join(test_path, dirpath_picture.split('\\')[-2])
            isExists = os.path.exists(path)
        ## 若0、1、2.。。文件夹不存在，则先创建文件夹。
            if not isExists:
                os.makedirs(path)
                print(dirpath_picture.split('\\')[-2]+' test set 创建成功')
        ## 更改文件的存储路径。
            try:
                img.save(os.path.join(path, dirpath_picture.split('\\')[-1]))
            except:
                pass
        
        ## 将后三分之二的图片移动到training-images文件夹下的对应文件夹下。
        for dirpath_picture in FileList[len(FileList)//10:]: ## dirpath_picture是图片的相对路径.\\result\\0\\picture.png。
            img = Image.open(dirpath_picture)
            ## 判断文件夹是否已经创建。
            path = os.path.join(train_path, dirpath_picture.split('\\')[-2])
            isExists = os.path.exists(path)
        ## 若0、1、2.。。文件夹不存在，则先创建文件夹。
            if not isExists:
                os.makedirs(path)
                print(dirpath_picture.split('\\')[-2]+' train set 创建成功')
        ## 更改文件的存储路径。
            try:
                img.save(os.path.join(path, dirpath_picture.split('\\')[-1]))
            except:
                pass
if __name__ == '__main__':

    resultDir = './data/dataset'
    test_path = './data/test'
    train_path = './data/train'
    split_dataset(resultDir, test_path, train_path)

