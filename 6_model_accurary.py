# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 17:14:34 2020

@author: ethan
"""

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model
from sklearn.metrics import accuracy_score
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True #这一样一定要有否则模型预测时会出错

from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score #全部实际患病的人员中，有多少被正确的检测为患病
from sklearn.metrics import f1_score
from keras.applications.imagenet_utils import decode_predictions
import time
import random
#%%
# Testing model on test data to evaluate

test_datagen = ImageDataGenerator(rescale=(1./255))

validation_generator = test_datagen.flow_from_directory(
        'D:\\data\\classification2\\test',
        target_size=(224,224),
        batch_size=32,
        shuffle = False,
        class_mode='categorical')

validation2_generator = test_datagen.flow_from_directory(
        'D:\\data\\classification2\\test2',
        target_size=(224,224),
        batch_size=32,
        shuffle = False,
        class_mode='categorical')


#%%模型加载
model = load_model('D:\\Project\\image_classification\\TensorFlow\\saved_models\\TF_1_%s_model.008_0.9565_class2.h5', custom_objects={'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D})
#%%
test_generator = validation2_generator
y_pred = model.predict_generator(test_generator, verbose=1)
y_pred= np.argmax(y_pred, axis=1)

#%%
y_true = test_generator.classes #每张图片的类别

class_indices = test_generator.class_indices #分类的所有名字
#%%
print(accuracy_score(y_true, y_pred)) #%%我们不能容忍假阴性，意味着病人诊断为健康。所以计算精度，我们要避开假阴性数据。
#%% 每一类的名字
print(classification_report(y_true, y_pred,target_names = list(class_indices)))
#%% 混淆矩阵
# Making the Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
#%%
recall = recall_score(y_true, y_pred,average='weighted')
print(recall)
#%%
f1 = f1_score(y_true, y_pred,average='weighted')
print(f1)
#%% 查看每一张图预测情况和实际情况
class_name = list(class_indices)
path = 'D:\\data\\classification2\\test2'
pathss = []
for root, dirss, files  in os.walk(path):
    # print(files)
    path = [os.path.join(root, name) for name in files]
    pathss.extend(path)

for i in range(len(pathss))[:]:
    
    img = image.load_img(pathss[i], target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    
    y_pred = model.predict(x)
    y_pred= int(np.argmax(y_pred, axis=1))
    # print('Real label: ',y_true[i],class_name[y_true[i]])
    # print('Predicted:', y_pred,class_name[y_pred])
    if y_true[i] != y_pred:
    #     os.remove(pathss[i])
        print(pathss[i])
        print(class_name[y_true[i]] + ' ' + class_name[y_pred])
        
#%% Top-k 准确度
k = 5 #top-5
#以下是计算方法
max_k_preds = y_pred.argsort(axis=1)[:, -k:][:, ::-1] #得到top-k label
match_array = np.logical_or.reduce(max_k_preds==y_true.reshape(len(y_true),1), axis=1) #得到匹配结果a_real = np.array([[1], [2], [1], [3]])

topk_acc_score = match_array.sum() / match_array.shape[0]
print(topk_acc_score)

# top1 = 0.0
# top5 = 0.0    
# class_probs = model.predict(x)
# for i, l in enumerate(labels):
#     class_prob = class_probs[i]
#     top_values = (-class_prob).argsort()[:5]
#     if top_values[0] == l:
#         top1 += 1.0
#     if np.isin(np.array([l]), top_values):
#         top5 += 1.0

# print("top1 acc", top1/len(labels))
# print("top1 acc", top5/len(labels))
#%% Top-k 查看每个图片单个情况
k = 3
for i in range(len(pathss)):
    # print(i)
    img = image.load_img(pathss[i], target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    
    y_preds = model.predict(x)
    
    arr = np.array(y_preds)
    top_k_idx=arr.argsort()[0][::-1][0:k]
    
    y_true_index = y_true[i]
    if y_true_index in top_k_idx:
        print(str(y_true_index) + ' ' + str(top_k_idx) + '  ' + 'in')
    else:
        print(str(y_true_index) + ' ' + str(top_k_idx) + '  ' + 'not in')
#%%可视化 一次12张图

import random

w=60
h=40
fig=plt.figure(figsize=(40, 30))
columns = 4
rows = 3    
for i in range(len(pathss))[:12]:
    print(i)
    img = image.load_img(pathss[i], target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    
    y_pred = model.predict(x)
    y_pred= int(np.argmax(y_pred, axis=1))
    # print('Real label: ',y_true[i],class_name[y_true[i]])
    # print('Predicted:', y_pred,class_name[y_pred])
          
    ax = fig.add_subplot(rows, columns, i+1)
    ax.set_title("Predicted result:" + class_name[y_true[i]]
                        +"\n"+"Actual result: "+ class_name[y_pred])
    plt.imshow(img)
    
plt.show()

#%% 中文显示 
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
w=60
h=40
fig=plt.figure(figsize=(40, 30))
columns = 4
rows = 3
j = 0  
for i in random.sample(range(len(pathss)), columns * rows): #随机取12张
    
    print(j)
    img = image.load_img(pathss[i], target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    
    y_pred = model.predict(x)
    y_pred= int(np.argmax(y_pred, axis=1))
    print('Real label: ',y_true[i],class_name[y_true[i]])
    print('Predicted:', y_pred,class_name[y_pred])
          
    ax = fig.add_subplot(rows, columns, j+1)
    # j = j + 1
    ax.set_title("Predicted result:" + str(class_name[y_true[i]])
                        +"\n"+"Actual result: "+ str(class_name[y_pred]))
    plt.imshow(img)
    
    j = j + 1

plt.show()