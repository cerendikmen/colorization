import numpy as np
from data_utils import *
from pathlib import Path
from extract_cifar10 import *
from skimage import color
from tempfile import TemporaryFile

# Read data
path = str(Path().absolute().parent)
path += '/cifar-10-python/cifar-10-batches-py/'
no_of_batches = 1
train_rgb, train_labels, test_rgb, test_labels = load_cifar10_data(path, no_of_batches) #HARA's EDIT PENDING REVIEW

# Convert data from RGB to LAB
N = len(train_rgb) #HARA'S EDIT PENDING REVIEW
train_lab = np.ndarray(shape=(N, 32, 32, 3), dtype=np.int8, order='C') #HARA: THIS SEEMS TO BE DUPLICATE OF LINE 32 IN extract_cifar10.py
for n in range(0, N):
	if n%1000==0:
		print(n,' images converted to LAB.')	
	train_lab[n] = color.rgb2lab(train_rgb[n])

test_lab = np.ndarray(shape=test_rgb.shape, dtype=np.int8, order='C')
for i in range(0, len(test_rgb)):
	test_lab[i] = color.rgb2lab(test_rgb[i])


#min_a, min_b, max_a, max_b = get_ab_ranges(train_lab)

'''
min_a: -86
max_a: 98
min_b: -107
max_b: 94

Intervals go 
a: -95 - 105
b: -115 - 95

'''
# get bins
grid_size = 10
inf_a = -95
sup_a = 105
bins_a = int((sup_a-inf_a)/grid_size)
inf_b = -115
sup_b = 95
bins_b = int((sup_b-inf_b)/grid_size)

qmat, bin_indices =  imgs2qmat(train_lab, grid_size, inf_a, sup_a, inf_b, sup_b)
lab_bins = get_lab_bins(qmat, bin_indices)
# Q = 241
# save to file
outfile = TemporaryFile()
np.save('lab_bins', lab_bins)

