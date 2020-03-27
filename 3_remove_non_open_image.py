# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:27:27 2019

@author: fg010
"""
 #%%
##########################################################################
 
import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
from PIL import Image
 
##########################################################################
 
 
def load_labels_file(filename,labels_num=1,shuffle=False):
    '''
    载图txt文件，文件中每行为一个图片信息，且以空格隔开：图像路径 标签1 标签2，如：test_image/1.jpg 0 2
    :param filename:
    :param labels_num :labels个数
    :param shuffle :是否打乱顺序
    :return:images type->list
    :return:labels type->list
    '''
    images=[]
    labels=[]
    with open(filename) as f:
        lines_list=f.readlines()
        if shuffle:
            random.shuffle(lines_list)
 
        for lines in lines_list:
            line=lines.rstrip().split(' ')
            label=[]
            for i in range(labels_num):
                label.append(int(line[i+1]))
            images.append(line[0])
            labels.append(label)
    return images,labels
 
def read_image(filename, resize_height, resize_width,normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''
    bgr_image = cv2.imdecode(np.fromfile(filename,dtype=np.uint8),1)

    if len(bgr_image.shape)==2:#若是灰度图则转为三通道
        print("Warning:gray image",filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
 
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)#将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height>0 and resize_width>0:
        rgb_image=cv2.resize(rgb_image,(resize_width,resize_height))
    rgb_image=np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image=rgb_image/255.0
    return rgb_image
 
def create_records(image_dir,file, resize_height, resize_width,shuffle,log=100):
    '''
    :param image_dir:原始图像的目录
    :param file:输入保存图片信息的txt文件(image_dir+file构成图片的路径)
    :param resize_height:
    :param resize_width:
    PS:当resize_height或者resize_width=0是,不执行resize
    :param shuffle:是否打乱顺序
    :param log:log信息打印间隔
    '''
    # 加载文件,仅获取一个label
    images_list, labels_list=load_labels_file(file,1,shuffle)
    error = 0
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):
        image_path=os.path.join(image_dir,images_list[i])
        if not os.path.exists(image_path):
            print('Err:no image',image_path)
            continue
        
        try:
            image = read_image(image_path, resize_height, resize_width)
        except:
            error += 1
            print('Err:unopen image',image_path)
            try:
                os.remove(image_path)  #删除不需要的文件
            except:
                continue
            continue
        image_raw = image.tostring()
        if i%log==0 or i==len(images_list)-1:
            print('------------processing:%d-th------------' % (i))
            print('current image_path=%s' % (image_path),'shape:{}'.format(image.shape),'labels:{}'.format(labels))
        label=labels[0]
    print('读取失败图片数： ', error)

#%%
 
if __name__ == '__main__':
    # 参数设置
    resize_height = 224  # 指定存储图片高度 224
    resize_width = 224  # 指定存储图片宽度 224
    shuffle=True
    log=3000
    
    # 删除train打不开的图
    image_dir= './data/train'
    train_labels = './data/train.txt' # 图片路径
    create_records(image_dir,train_labels, resize_height, resize_width,shuffle,log)

    # 删除test打不开的图
    shuffle=False
    image_dir='./data/test'
    val_labels = './data/test.txt'  # 图片路径
    create_records(image_dir,val_labels, resize_height, resize_width,shuffle,log)

