# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 14:44:36 2019

@author: fg010
"""
import os
import cv2
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import glob
"""
该过程可选
需要预先复制一份train set到同一个文件夹下，取名为train_data_augmentation
"""
def Data_Augmentation(path,save_path):
    all_images = glob.glob(path + '\\*.jpg')# 获取目录下所有图片路径

    for path in all_images:
        name = os.path.basename(path)[:-4]
        try:
            images = cv2.imdecode(np.fromfile(path,dtype=np.uint8),1)
        except:
            continue
        images = [images, images, images]# 数据量变成3倍
        sometimes = lambda aug: iaa.Sometimes(0.5, aug) # 定义一个lambda表达式，以p=0.5的概率去执行sometimes传递的图像增强

        seq =iaa.Sequential(
            [
            iaa.Fliplr(0.5), # horizontal flips
            iaa.Flipud(0.5),
            sometimes(iaa.Crop(percent=(0, 0.1))),
            sometimes(iaa.Affine( # 部分图像做仿射变换
            scale = {'x':(0.8,1.2),'y':(0.8,1.2)},# 图像缩放为80%到120%
            translate_percent={'x':(-0.2,0.2),'y':(-0.2,0.2)},# 平移±20%
            rotate=(-20,20),# 旋转±20度
            shear=(-16,16),#剪切变换±16度（矩形变平行四边形）
            cval=(0,255),# 全白全黑填充
#            mode=ia.ALLL# 定义填充图像外区域的方法
                    )),
#        使用下面的0个到2个之间的方法增强图像
#            iaa.SomeOf((0,2),
#                [
#                iaa.Sharpen(alpha=(0,0.3),lightness=(0.9,1.1)),#锐化处理
#                # 加入高斯噪声
#                iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5),
#                iaa.Add((-10,10),per_channel=0.5),# 每个像素随机加减-10到10之间的数
#                iaa.Multiply((0.8,1.2),per_channel=0.5),# 像素乘上0.5或者1.5之间的数字
#                # 将整个图像的对比度变为原来的一半或者二倍
#                iaa.ContrastNormalization((0.5,2.0),per_channel=0.5),
#                ],
#                random_order=False)
                ],
            random_order=True # 随机的顺序把这些操作用在图像上
            )
        try:    
            images_aug = seq.augment_images(images)# 应用数据增强
        except:
            continue

        c = 1
        for each in images_aug:
            cv2.imencode('.jpg',each)[1].tofile(save_path + '\\%s%s.jpg'%(name, c))
#        cv2.imencode('.jpg',each)[1].tofile('C:\\Users\\fg010\\Desktop\\%s%s.jpg'%(name, c), each)# 增强图片保存到指定路径
#        cv2.imwrite('C:\\Users\\fg010\\Desktop\\荸荠\\%s%s.jpg'%(name, c), each)
            c+=1
    # ia.imshow(np.hstack(images_aug))# 显示增强图片
    print('增强图片完成')
#%% 
train_dir = './data/train'
i = 0
for dirname in os.listdir(train_dir): ## 获取当前路径下的所有文件夹及文件名。 dirname是0、1、2.。。。文件夹的名字。
    path = train_dir + '\\' + dirname
    print(path, i)
    Data_Augmentation(path,path)
    i += 1