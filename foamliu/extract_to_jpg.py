# this script extract cifar batch files to a folder with .jpg
import numpy as np
from pathlib import Path
#from skimage import color
#from scipy.ndimage import gaussian_filter
from extract_cifar10 import *
#from data_utils import *
import scipy.misc
#from matplotlib.pyplot import imread

path = str(Path().absolute())
path += '/cifar-10-python/cifar-10-batches-py/'
print('\n',path)
no_of_batches = 5
train_rgb, train_labels, test_rgb, test_labels = load_cifar10_data(path, no_of_batches)
#tiny_ndarray = train_rgb[0:10,:,:,:]

file1 = open("train_names.txt","w")
file2 = open("valid_names.txt","w")
L1 = []
L2 = []

for i in range(0, len(train_rgb)):
	if (i+1)%1000 == 0:
		print((i+1), ' training images saved')
	file_name = '{}_train_image.jpg\n'.format(i)
	L1.append(file_name)
	scipy.misc.imsave('cifar-10/{}_train_image.jpg'.format(i), train_rgb[i,:,:,:])

for i in range(0, len(test_rgb)):
	if (i+1)%1000 == 0:
		print((i+1), ' testing images saved')
	file_name = '{}_test_image.jpg\n'.format(i)
	L2.append(file_name)
	scipy.misc.imsave('cifar-10/{}_test_image.jpg'.format(i), test_rgb[i,:,:,:])

file1.writelines(L1) 
file2.writelines(L2) 
file1.close() #to change file access modes 
