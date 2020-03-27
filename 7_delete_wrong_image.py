# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:32:36 2020

@author: ethan
"""

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import models
# from VESI import layers
from keras.applications.imagenet_utils import decode_predictions

from keras.models import load_model

model = load_model('D:\\Project\\image_classification\\TensorFlow\\saved_models\\TF_1_%s_model.008_0.9565_class2.h5', custom_objects={'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D})
#%%
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
test_datagen = ImageDataGenerator(rescale=(1./255))

test_generator = test_datagen.flow_from_directory(
        'D:\\data\\classification2\\test2',
        target_size=(224,224),
        batch_size=1,
        shuffle = False,
        class_mode='categorical')

y_pred = model.predict_generator(test_generator, verbose=1)
#%%
# y_pred= np.argmax(y_pred, axis=1)

y_true = test_generator.classes
#%%
# for i in range(100):
#     test_generator.next()
test_generator.links()
#%%
path = 'D:\\data\\classification2\\test2'
pathss = []
for root, dirss, files  in os.walk(path):
    # print(files)
    path = [os.path.join(root, name) for name in files]
    pathss.extend(path)
j = 0
k = 3
for i in range(len(pathss)):
    print(i)
    img = image.load_img(pathss[i], target_size=(224, 224))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x, axis=0)
    
    y_preds = model.predict(x)
    
    y_pred= int(np.argmax(y_pred, axis=1))
    print('Real label: ',y_true[i])
    print('Predicted:', y_pred)
    if y_true[i] != y_pred:
        os.remove(pathss[i])
        print(pathss[i])
    j += 1