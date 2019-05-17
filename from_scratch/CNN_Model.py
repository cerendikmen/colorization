#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Built on Python 3.6.8
import numpy as np #numpy version 1.16.3
import tensorflow as tf #tensorflow-gpu version 1.13.1
from tensorflow import keras


# In[2]:


#Import Data
train_images = np.load('./cifar-10-npy/train_data.npy')
train_labels = np.load('./cifar-10-npy/train_labels_1hot.npy')
test_images = np.load('./cifar-10-npy/test_data.npy')
test_labels = np.load('./cifar-10-npy/test_labels_1hot.npy')


# In[3]:


#Preprocessing as described in 0-center data_l channel in CAFFE
train_images = train_images - 50
test_images = test_images - 50


# In[4]:


#Reduce Size of Data Set to Prevent Memory Overflow and Speed Up Training
no_of_labels = int(train_labels.max() + 1)

train_images = np.reshape(train_images, newshape=(len(train_images),len(train_images[0]),len(train_images[0][0]),1))
train_labels = np.reshape(train_labels, newshape=(len(train_images),len(train_images[0]),len(train_images[0][0]),1))

#On google server, see if training can be done on all images
#train_images = train_images[0:200]
#train_labels = train_labels[0:200]

test_images = np.reshape(test_images, newshape=(len(test_images),len(test_images[0]),len(test_images[0][0]),1))
test_labels = np.reshape(test_labels, newshape=(len(test_images),len(test_images[0]),len(test_images[0][0]),1))

test_images = test_images[0:40]
test_labels = test_labels[0:40]


# In[5]:

'''
model = keras.Sequential([
    keras.layers.Conv2D(64, 3, padding='same', input_shape=(32, 32, 1), activation=tf.nn.relu), #bw_conv1_1
    keras.layers.BatchNormalization(), #conv1_2norm
    keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=1), #conv8_3
    keras.layers.Conv2D(no_of_labels, 1, padding='same', strides=1, dilation_rate=1), #convolve to labels
])

'''
model = keras.Sequential([
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(64, 3, padding='same', input_shape=(32, 32, 1), activation=tf.nn.relu), #bw_conv1_1
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(64, 3, padding='same', activation=tf.nn.relu, strides=2), #conv1_2
    keras.layers.BatchNormalization(), #conv1_2norm
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu), #conv2_1
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(128, 3, padding='same', activation=tf.nn.relu, strides=2), #conv2_2
    keras.layers.BatchNormalization(), #conv2_2norm
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu), #conv3_1
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu), #conv3_2
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, strides=2), #conv3_3
    keras.layers.BatchNormalization(), #conv3_3norm
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=1), #conv4_1
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=1), #conv4_2
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=1), #conv4_3
    keras.layers.BatchNormalization(), #conv4_3norm
    keras.layers.ZeroPadding2D(padding=2),#need padding of 2
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=2), #conv5_1
    keras.layers.ZeroPadding2D(padding=2),#need padding of 2
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=2), #conv5_2
    keras.layers.ZeroPadding2D(padding=2),#need padding of 2
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=2), #conv5_3
    keras.layers.BatchNormalization(), #conv5_3norm
    keras.layers.ZeroPadding2D(padding=2),#need padding of 2
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=2), #conv6_1
    keras.layers.ZeroPadding2D(padding=2),#need padding of 2
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=2), #conv6_2
    keras.layers.ZeroPadding2D(padding=2),#need padding of 2
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=2), #conv6_3
    keras.layers.BatchNormalization(), #conv6_3norm
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=1), #conv7_1
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=1), #conv7_2
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(512, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=1), #conv7_3
    keras.layers.BatchNormalization(), #conv7_3norm
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2DTranspose(256, 4, padding='same', activation=tf.nn.relu, strides=2, dilation_rate=1), #conv8_1
    keras.layers.ZeroPadding2D(padding=1),
    keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=1), #conv8_2
    #keras.layers.ZeroPadding2D(padding=1),
    #keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, strides=1, dilation_rate=1), #conv8_3
    keras.layers.Conv2D(256, 3, padding='same', activation=tf.nn.relu, strides=3, dilation_rate=1), #modified conv8_3
    keras.layers.Conv2D(no_of_labels, 1, padding='same', strides=1, dilation_rate=1) #convolve to labels
])

# In[6]:


model.compile(optimizer='SGD', 
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[7]:


no_of_epochs = 1 #HYPERPARAMETER
for e in range (0, no_of_epochs):
    print("Starting epoch #", e+1)
    for n in range(0, len(train_labels)):
        train_images_n = np.reshape(train_images[n], newshape=(1,len(train_images[n]),len(train_images[n][0]),1))
        train_labels_q = np.zeros((1,len(train_images[n]),len(train_images[n][0]),no_of_labels))
        for i in range(0, len(train_labels[n])):
            for j in range(0, len(train_labels[n][0])):
                label = train_labels[0][i][j][0]
                train_labels_q[0][i][j][int(label)] = 1
        model.fit(train_images_n, train_labels_q, epochs=1, batch_size=1, verbose=0)
		if ((n+1)%1000 == 0):
			print("Trained on ", n+1, " images")
    print("Finished epoch #", e+1)


# In[8]:


test_labels_q = np.zeros((len(test_images),len(test_images[0]),len(test_images[0][0]),no_of_labels))

for n in range(0, len(test_labels)):
    for i in range(0, len(test_labels[0])):
        for j in range(0, len(test_labels[0][0])):
            label = test_labels[n][i][j][0]
            test_labels_q[n][i][j][int(label)] = 1

test_loss, test_acc = model.evaluate(test_images, test_labels_q)
print('Test accuracy:', test_acc)


# In[9]:


prediction = model.predict(test_images)
prediction = np.reshape(prediction, newshape=(40,32,32,no_of_labels))
prediction = prediction.argmax(axis=3)
prediction = np.reshape(prediction, newshape=(40,32,32,1))


# In[10]:


np.save('./cifar-10-npy/test_labels_predictions', prediction)
np.save('./cifar-10-npy/test_data_predictions', test_images)

