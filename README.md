# 图片分类image_classification
使用百度爬虫对65种果蔬就行分类

使用Keras MobilenetV2进行分类

数据增强可选

### 环境
Anaconda Python=3.6 Keras=2.2.4实现

### 步骤
按照0,1,2,3,4,5,6进行

##### 文件路径可能需要设置

### 0. 爬虫
`0_baidu_image.py`（可选） 

需要输入关键词，这里设置为在` ./data/label.csv` 中，把关键词放在在该`csv`表格中即可

根据数量需要修改以下地方，注意，一页为60张图片左右
```python
page_begin = 0 #开始页码
page_number = 0  #最终页码
image_number = 0 #默认0为60张，1为1张，2为2张，以此类推
```

为多线程爬取，设置为5~10比较合适

按照给与的CSV文件进行爬虫下载，可能需要人工剔除部分干扰图片

### 1.划分训练集检测集
`1_split_dataset.py`

### 2. 生成带label 的图片文件路径
`2_create_labels_files.py`

### 3. 删除一些带不开的图片
`3_remove_non_open_image.py`

有一些打不开的图片，如果不删除训练会报错

### 4. 样本增强（包需要安装）

`4_Data_Augmentation.py`（可选）

不同于实时增强，该文件将增强的图片保存在本地

### 5. 训练 
`5_trian.py`

* 该模型使用 MobilenetV2，准确率为95%左右，如果使用其他模型只需要在`model.add()`更换即可。如果遇到利用keras的InceptionV3、ResNet50模型做迁移学习训练集和验证集的准确率相差很大的问题，比如验证集准确率很小，参考[这里](https://blog.csdn.net/zjn295771349/article/details/86355874)
* 该文件为输出图片大小为224*224, batch size大小为32（GPU 8G内存）
* 模型训练只对最后的输出层做出修改，其他层不可改，如果其他层数需要更新，请将`model.trainable `设置为`True`
* 数据预处理只将其归一化，然后使用`flow_from_directory`分批读入内存。在`flow_from_directory`中，需要将图片打乱，在`train_generator`中将`shuffle=True`，`validation_generator`设置为`False`。
* 并进行学习速率更新衰减，设置为一定轮数后减至原来的![](http://latex.codecogs.com/gif.latex?\\frac{1}{10})
，`lr_schedule`函数和`keras.callbacks.LearningRateScheduler`函数为此服务
* 使用`keras.callbacks.CSVLogger`，将训练数据储存在`result.csv`中
* 使用`ModelCheckpoint`，定义存储模型的路径，训练中可视化

训练，使用` model.fit_generator`，在里面设置的参数有很多，其中`steps_per_epoch`为训练集大小除以`batch_size`。 注意：`workers`为CPU使用数量，更具CPU设置为最大，4核设置为4,8核设置为8，超过也不会报错，尽量往大的写。

### 6. 检测
`6_model_accurary.py`

读取模型时可能会有错误（现在没错），原因是卷基层导致的，具体错误百度或谷歌一下

训练完成后，需要进行验证，并可视化

混淆矩阵判定测试集中模型优劣，

### 7.  使用该模型删除在其他数据集不可用的图片 实例
`7_delete_wrong_image.py`
