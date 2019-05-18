#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Built on Python 3.6.8
import numpy as np #numpy version 1.16.3
from skimage import color #skimage version 0.15.0

from extract_cifar10 import *


# In[3]:


# Read data
path = '../cifar-10-python/cifar-10-batches-py'
no_of_batches = 5 #Hyperparameter: how much training data
train_rgb, _, test_rgb, _ = load_cifar10_data(path, no_of_batches) #class labels are discarded


# In[4]:


# Convert data from RGB to LAB
N = len(train_rgb)
train_lab = np.ndarray(train_rgb.shape)
for n in range(0, N):
    train_lab[n] = color.rgb2lab(train_rgb[n])
    if (n+1)%1000==0:
        print((n+1),' training images converted to LAB.')

test_lab = np.ndarray(test_rgb.shape)
for i in range(0, len(test_rgb)):
    test_lab[i] = color.rgb2lab(test_rgb[i])
    if (i+1)%1000==0:
        print((i+1),' testing images converted to LAB.')


# In[5]:


#Find the edges of the ab color space
inf_a = train_lab[:,:,:,1].min()
sup_a = train_lab[:,:,:,1].max()
inf_b = train_lab[:,:,:,2].min()
sup_b = train_lab[:,:,:,2].max()
print('a-range: ', inf_a, sup_a, "; b-range: ", inf_b, sup_b)


# In[6]:


# bins are inclusive from below but not from above, e.g. -5 to 4, or 15 to 24
def imgs2qmat(train_lab, grid_size, inf_a, sup_a, inf_b, sup_b):
    awidth = sup_a-inf_a
    bwidth = sup_b-inf_b
    bins_a = int(awidth//grid_size + (awidth % grid_size > 0)) #ceiling rounding without importing math package
    bins_b = int(bwidth//grid_size + (bwidth % grid_size > 0))
    
    qmat = np.zeros((bins_a, bins_b))
    
    train_lab = train_lab[:,:,:,1:3]
    inf_array = np.array([inf_a,inf_b])
    train_lab = train_lab - inf_array
    train_lab = train_lab / grid_size
    
    for n in range(0, len(train_lab)):
        for i in range(0, 32):
            for j in range(0, 32):
                k = int(np.floor(train_lab[n][i][j][0]))
                l = int(np.floor(train_lab[n][i][j][1]))
                qmat[k][l] += 1
        if (n+1)%1000==0:
            print((n+1),' images converted to matrix.')
    return qmat, train_lab


# In[7]:


#convert to bins
grid_size = 10 #Hyperparameter: dimension of bin (in ab space, so grid size of 10 equates to 100 pixels in ab space)
qmat, bin_indices =  imgs2qmat(train_lab, grid_size, inf_a, sup_a, inf_b, sup_b)
#These functions needed to create Q vector per pixel, but for now we are using one hot encoding
#lab_bins = get_lab_bins(qmat, bin_indices)


# In[8]:


#Create Dictionaries that Convert from Bin# to AB
q2ab = {}
ab2q = {}
qid = 0
for a in range(0,len(qmat)):
    for b in range(0,len(qmat[0])):
        if qmat[a][b] > 0:
            q2ab[qid] = (a,b)
            ab2q[(a,b)] = qid
            qid = qid+1
print('Number of bins: ', qid)


# In[9]:


train_labels = np.zeros((len(bin_indices),len(bin_indices[0]),len(bin_indices[0][0])))
for n in range(0,len(bin_indices)):
    for i in range(0,len(bin_indices[0])):
        for j in range (0,len(bin_indices[0][0])):
            train_labels[n][i][j] = ab2q[(int(bin_indices[n][i][j][0]), int(bin_indices[n][i][j][1]))]
    if (n+1)%1000==0:
        print((n+1),' training images received bin classification.')
            
train_data = train_lab[:,:,:,0]


# In[10]:


test_ab = test_lab[:,:,:,1:3]
inf_array = np.array([inf_a,inf_b])
test_ab = test_ab - inf_array
test_ab = test_ab / grid_size
    
test_labels = np.zeros((len(test_ab),len(test_ab[0]),len(test_ab[0][0])))
for n in range(0,len(test_ab)):
    for i in range(0,len(test_ab[0])):
        for j in range (0,len(test_ab[0][0])):          
            #WARNING: Results in error if the training sample does not contain a color found in the test sample
            test_labels[n][i][j] = ab2q[(int(test_ab[n][i][j][0]), int(test_ab[n][i][j][1]))]
    if (n+1)%1000==0:
        print((n+1),' test images received bin classification.')        
test_data = test_lab[:,:,:,0]


# In[11]:


# save to file
np.save('./cifar-10-npy/train_labels_1hot', train_labels)
np.save('./cifar-10-npy/train_data', train_data)
np.save('./cifar-10-npy/test_labels_1hot', test_labels)
np.save('./cifar-10-npy/test_data', test_data)
np.save('./cifar-10-npy/bins_to_ab_dictionary', q2ab)
np.save('./cifar-10-npy/binning_parameters', np.array([inf_a,inf_b, grid_size]))

