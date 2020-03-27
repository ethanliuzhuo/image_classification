# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 10:49:26 2019

@author: fg010
"""
#%%
from __future__ import print_function,absolute_import, division, print_function, unicode_literals
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
import numpy as np
import os
import random

import tensorflow as tf
from create_tf_record import *
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from keras import regularizers
from keras.models import load_model
#%%

model = load_model("D:\\OneDrive\\工作\\富港万嘉\\TensorFlow\\saved_models\\TL_0_%s_model.036_0.8795.h5")
# x_test = np.load('x_test_64.npy')
# y_test = np.load('y_test.npy')
#%%
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
    # show_image("src resize image",image)
    return rgb_image
 
def create_records(image_dir,file, resize_height, resize_width,shuffle = False,log=100,normalization=True):
    '''
    实现将图像原始数据,label,长,宽等信息保存为record文件
    注意:读取的图像数据默认是uint8,再转为tf的字符串型BytesList保存,解析请需要根据需要转换类型
    :param image_dir:原始图像的目录
    :param file:输入保存图片信息的txt文件(image_dir+file构成图片的路径)
    :param output_record_dir:保存record文件的路径
    :param resize_height:
    :param resize_width:
    PS:当resize_height或者resize_width=0是,不执行resize
    :param shuffle:是否打乱顺序
    :param log:log信息打印间隔
    '''
    # 加载文件,仅获取一个label
    images_list, labels_list=load_labels_file(file,1,shuffle)
    error = 0
#    writer = tf.python_io.TFRecordWriter(output_record_dir)
    images = []
    images_labels = []
    for i, [image_name, labels] in enumerate(zip(images_list, labels_list)):
#        print(i)
#        print([image_name, labels])
        image_path=os.path.join(image_dir,images_list[i])
#        print(image_path)
        if not os.path.exists(image_path):
            # print('Err:no image',image_path)
            continue
#        
        try:
            image = read_image(image_path, resize_height, resize_width)
        except:
            error += 1
            continue
        
        # if i%log==0 or i==len(images_list)-1:
        #     print('------------processing:%d-th------------' % (i))
        #     print('current image_path=%s' % (image_path),'shape:{}'.format(image.shape),'labels:{}'.format(labels))
        # 这里仅保存一个label,多label适当增加"'label': _int64_feature(label)"项
        label=labels[0]
        images += [image]
        images_labels += [label]
    # print('读取失败图片数： ', error)
    
    if normalization == True:
        images = np.array(images)/255.0
        images_labels = np.array(images_labels)
        # np.save(save_path + 'images.npy', arr=images)
        # np.save(save_path + 'labels.npy', arr=images_labels)
    else:
        images = np.array(images)
        images_labels = np.array(images_labels)
        # np.save(save_path + 'images.npy', arr=images)
        # np.save(save_path + 'labels.npy', arr=images_labels)
    
    return images,images_labels
#%%
# import os
# import os.path
# import pandas as pd

 
# def write_txt(content, filename, mode='w'):
#     """保存txt数据
#     :param content:需要保存的数据,type->list
#     :param filename:文件名
#     :param mode:读写模式:'w' or 'a'
#     :return: void
#     """
#     with open(filename, mode) as f:
#         for line in content:
#             str_line = ""
#             for col, data in enumerate(line):
#                 if not col == len(line) - 1:
#                     # 以空格作为分隔符
#                     str_line = str_line + str(data) + " "
#                 else:
#                     # 每行最后一个数据用换行符“\n”
#                     str_line = str_line + str(data) + "\n"
#             f.write(str_line)

# def get_files_list(dir,lable_dir):
#     '''
#     实现遍历dir目录下,所有文件(包含子文件夹的文件)
#     :param dir:指定文件夹目录
#     :return:包含所有文件的列表->list
#     '''
#     # parent:父目录, filenames:该目录下所有文件夹,filenames:该目录下的文件名
#     files_list = []
#     label_data = pd.read_csv(lable_dir,header=0)
#     for parent, dirnames, filenames in os.walk(dir):
#         for filename in filenames:
#             # print("parent is: " + parent)
#             # print("filename is: " + filename)
#             # print(os.path.join(parent, filename))  # 输出rootdir路径下所有文件（包含子文件）信息
#             curr_file=parent.split(os.sep)[-1]
#             labels = label_data[label_data['物品'] == curr_file].index.tolist()[0]
#             files_list.append([os.path.join(curr_file, filename),labels])#文件夹下路径
#             # files_list.append([os.path.join(parent,filename),labels])#绝对路径
#             print()
#     return files_list 
# lable_dir = 'D:\\data\\classification\\label.csv'
    
# train_dir = 'D:\\data\\classification\\test2'
#     # train_txt='D:\\data\\classification\\train.txt'
# train_txt='D:\\data\\classification\\test2.txt'
# train_data = get_files_list(train_dir,lable_dir)
# write_txt(train_data,train_txt,mode='w')
#%%
name = ['丝瓜', '中华猕猴桃', '冬瓜', '南瓜', '哈密瓜', '大白菜', '大蒜', '快圆茄', '木瓜', '杨桃', '杨梅', '枣', '柚', '柠檬', '柿', '桂圆', '桃', '桑葚', '梨', '椰子', '樱桃', '橙', '沙棘', '油麦菜', '洋葱', '甜椒', '番茄', '白萝卜', '百合', '秋葵', '紫皮大蒜', '细香葱', '胡萝卜', '节瓜', '芒果', '芥蓝', '芹菜', '苦瓜', '苹果', '茄子', '茼蒿', '草莓', '荔枝', '荷兰豆', '荸荠', '莴苣', '菠菜', '菠萝', '菠萝蜜', '葡萄', '藕', '西兰花', '西瓜', '豆角', '豌豆', '辣椒', '青萝卜', '韭菜', '韭黄', '香菜', '香蕉', '鳄梨', '黄瓜', '黄皮果', '黄豆芽']
dirs = 'D:\\data\\classification\\data_clean\\test_clean\\'
file = 'D:\\data\\classification\\data_clean\\test_clean.txt'
import numpy as np
from sklearn.metrics import classification_report

x_test,y_test = create_records(dirs + '丝瓜',file, resize_height=224, resize_width =224,shuffle = False,log=100)
# for i in name:
#     # print(dirs + i)
#     x_test,y_test = create_records(dirs + i,file, resize_height=224, resize_width =224,shuffle = False,log=100)
    # predict_test = model.predict(x_test)
    
#     predict = np.argmax(predict_test,axis=1)
#     # scores = model.evaluate(x_test, y_test, verbose=1)
#     # print(i)
#     # print('Test accuracy:', predict)
#     print(i+': '+(y_test == predict).sum()/len(predict))
#     # # open(os.path.join(dirs,i))
#     # print(os.path.join(dirs,i))
#%%
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])