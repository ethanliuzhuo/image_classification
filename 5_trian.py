
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D,Input
from keras.applications import MobileNetV2,ResNet50
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam,SGD
from keras.applications.resnet50 import ResNet50
from keras import models
# from VESI import layers
from keras import optimizers
from keras import regularizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,CSVLogger
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense


target_size = (224,224)
batch_size = 32
#模型
model = Sequential()
model.add(MobileNetV2(include_top = False,pooling='avg', weights = 'imagenet',input_shape=(224, 224, 3)))
model.trainable = False
model.add(Dense(66, activation = 'softmax'))
# model.layers[0].trainable = True #或者这个

# model = Sequential()
# model.add(MobileNetV2(include_top = False, weights = './saved_models/TF_2_%s_model.008.h5',input_shape=(224, 224, 3)))
# model.add(Dense(65, activation = 'softmax'))
# model.layers[0].trainable = True

#数据预处理
train_datagen = ImageDataGenerator(
        rescale=1./255)
        # shear_range=0.2, #实时图片增强
        # zoom_range=0.2,
        # horizontal_flip=True)
        # rotation_range=180,
        # width_shift_range=0.3,
        # height_shift_range=0.3,
        # zoom_range=0.3,
        # horizontal_flip=True)
        # vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
#图片分批读入
train_generator = train_datagen.flow_from_directory(
        'D:\\data\\classification2\\train',
        target_size=target_size,
        batch_size=batch_size,
        shuffle = True,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'D:\\data\\classification2\\test',
        target_size=target_size,
        batch_size=batch_size,
        shuffle = False,
        class_mode='categorical')
# 学习速率衰减
def lr_schedule(epoch):

    lr = 1e-2
    if epoch > 40:
        lr *= 0.5e-3
    elif epoch > 30:
        lr *= 1e-3
    elif epoch > 8:
        lr *= 1e-2
    elif epoch > 3:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr
#储存模型
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'TF_1_%s_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = ModelCheckpoint(filepath=filepath,
                              monitor='val_acc',
                              verbose=1,
                              period=1,
                              save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
#储存训练数据
logger = CSVLogger('./result.csv', separator=',', append=True)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                cooldown=0,
                                patience=5,
                                min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler,logger]
#优化器
sgd = SGD(lr=lr_schedule(0))

model.compile(loss='categorical_crossentropy',
                # optimizer=Adam(lr=lr_schedule(0)),
                optimizer=sgd,
              metrics=['accuracy'])

fit_history = model.fit_generator(train_generator,
                                  steps_per_epoch = 39477/batch_size,
                                  validation_data=validation_generator,
                                  validation_steps=4349/batch_size,
                                  epochs=200, verbose=1, workers=8,
                                  callbacks=callbacks)

#%% 训练可视化
def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))
    f, ax = plt.subplots(1,2, figsize=(14,6))
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.show()
plot_accuracy_and_loss(fit_history)