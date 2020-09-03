#!/usr/bin/env python
# coding: utf-8

# CNN classifier training
# Train a CNN classifier with two convolutional layers of filter size 3x3 (followed
# by relu and max-pooling with 2x2 kernels), and two fully-connected layers of
# equal size L. This classifier should be trainable on each of the 4 classification
# problems. (Look at the lecture slides for more details!) To adapt classifiers to
# a particular problem (from the four problems that are given), you can vary the
# following parameters that should be declared as ”constants” at the beginning
# of the program:
# • learning rate  (start with 0.01)
# • hidden layer size L (start with 50)
# • image width, height and channels (depends on problem)
# • number of training iterations (start with 50)
# Monitor the test error every T iterations! Today, do this by downloading
# the last layer activities of the CNN to numpy and compare these to the test
# targets (testl) you already have in numpy format. Compute the test error and
# the confusion matrix and display them

# In[1]:


import os,sys;
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np;
import tensorflow as tf;
import matplotlib.pyplot as plt;
import PIL.Image as Img;


# GLOBAL VARIBALES

# In[2]:


eps = 0.01
L = 50
imgH = 28
imgW = 28
imgC = 1
max_iter = 1000
nrClass = 10
path = r"C:\Users\damil\Desktop\MachineLearning\Class12\DataSet\mnist";
B = 50


# EXTRACT DATA

# In[3]:


def loadImgDir(path):
    fileList = os.listdir(path)
    imgObjs = []
    for f in sorted(fileList):
        if f.find(".png") != -1:
            of = path + "/" + f;
            o = Img.open(of); o.load();
            imgObjs.append(np.array(o).ravel());
    imgArr = np.array(imgObjs)
    
    #Convert to One-Hot
    numLabels = np.array([int(f.split("-")[0]) for f in fileList])
    oneHot = np.zeros([len(numLabels),10])
    for i in range (0, len(numLabels)):
        oneHot[i,numLabels[i]] = 1
    return imgArr, oneHot


# LOAD UP DATA 

# In[4]:


xData,tData = loadImgDir(path)


# SHUFFLE TRAIN DATA - INPLACE

# In[5]:


if True:
    indices = np.arange(0, xData.shape[0])
    np.random.shuffle(indices)
    xData = xData[indices,:]
    tData = tData[indices,:]


# SPLIT DATA CAUSE OF OVERFITTING

# In[6]:


nu = 0.8
splitIndex = int(xData.shape[0] * nu)
traind = xData[0:splitIndex,:]
trainl = tData[0:splitIndex,:]
testd = xData[splitIndex:,:]
testl = tData[splitIndex:,:]


# CREATE TENSORFLOW PLACEHOLDER

# In[7]:


inputData = tf.placeholder(dtype = tf.float32, shape = [None,imgH*imgW*imgC], name= "InputData")
targetData = tf.placeholder(dtype = tf.float32, shape = [None, nrClass], name =  "TargetData")


# STARTING DATA

# In[8]:


ao = tf.reshape(inputData, (-1,imgH,imgW,imgC))


# CONVOLUTION LAYERS

# CONVOLUTION LAYER 1

# In[9]:


nrFilters = 32 ; 
filterSize = 3; #3 x 3
conv1 = tf.layers.conv2d(ao,nrFilters, filterSize, activation = tf.nn.relu, name = "conv1")


# In[10]:


kernelSize = 2; #2 x 2
strides = 2;
a1 = tf.layers.max_pooling2d(conv1,kernelSize,strides, name = "MaxPolling1")


# CONVOLUTION LAYER 2

# In[11]:


nrFilters2 = 64;
filterSize2 = 3;
conv2 = tf.layers.conv2d(a1,nrFilters2,filterSize2, activation = tf.nn.relu, name = "Conv2")
print(conv2.shape)


# In[12]:


kernelSize2 = 2;
strides = 2;
a2 = tf.layers.max_pooling2d(conv2,kernelSize2,strides, name= "MaxPolling2")


# In[13]:


newShape = a2.shape
print(newShape)
newSize = int(newShape[1] * newShape[2] * newShape[3])
a2Flat = tf.reshape(a2, (-1, newSize))
print(newSize)


# START DNN LAYER

# In[14]:


W3 = tf.Variable(np.random.uniform(-0.01,0.01,[newSize, L]),dtype = tf.float32)
B3 = tf.Variable(np.random.uniform(-0.01,0.01,[L]), dtype = tf.float32)
a3 = tf.nn.relu_layer( a2Flat, W3, B3, "a3")


# In[15]:


W4 = tf.Variable(np.random.uniform(-0.01,0.01, [L,L]), dtype = tf.float32)
B4 = tf.Variable(np.random.uniform(-0.01,0.01,[1,L]), dtype = tf.float32)
a4  = tf.nn.relu(tf.matmul(a3,W4) + B4)


# In[22]:


W5 = tf.Variable(np.random.uniform(-0.01, 0.01,[L,nrClass]), dtype = tf.float32)
B5 = tf.Variable(np.random.uniform(-0.01,0.01, [nrClass]), dtype = tf.float32)
output = tf.nn.matmul(a4, W5) + B5


# LOSS FUNCTION

# In[ ]:


#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(targetData,outPut))
loss = tf.reduce_mean((targetData - output) ** 2)


# LOSS OPTIMIZER

# In[ ]:


adamOpt = tf.train.AdamOptimizer(learning_rate = eps)
optimizedLoss = adamOpt.minimize(loss)


# CLASSIFICATION ERROR

# In[ ]:


comp = tf.equal(tf.argmax(targetData,axis = 1), tf.argmax(output, axis = 1))
acc = tf.reduce_mean(tf.cast(comp,tf.float64))
ce = 1 - acc


# CREATE A SESSION

# In[ ]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# RUN ITERATION

# In[ ]:


for i in range(0,max_iter):
    trainBatch = traind[i*B : (i+1)*B]
    targetBatch = trainl[i*B : (i+1) *B]
    sess.run(optimizedLoss, feed_dict = { inputData : trainBatch, targetData : targetBatch})
    testErr = sess.run(ce, feed_dict = { inputData : testd, targetData : testl})
    if( i % 50 == 0):
        print(i, "Test Error is " , testErr)


# In[ ]:




