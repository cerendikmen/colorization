import numpy as np
from pathlib import Path
from extract_cifar10 import *
from skimage import io, color

# Read data
path = str(Path().absolute().parent)
path += '/cifar-10-python/cifar-10-batches-py/'
train_data, train_labels, test_data, test_labels = load_cifar10_data(path)

# Convert data from RGB to LAB
train_lab = np.ndarray(shape=train_data.shape, dtype=np.int8, order='C')
for n in range(0, len(train_data)):
	train_lab[n] = color.rgb2lab(train_data[n])
	#for i in range(0,32):
		#for j in range(0,32):
			#train_lab[n][i][j][1] -= 128
'''
test_lab = np.ndarray(shape=test_data.shape, dtype=np.uint8, order='C')
for i in range(0, len(test_data)):
	test_lab[i] = color.rgb2lab(test_data[i])
'''

# train_lab is a 50000 long 4d array storing images in LAB-space
min_a = 1000
min_b = 1000
max_a = -1000
max_b = -1000
for n in range(0, len(train_data)):
	if n%1000==0:
		print(n)
	for i in range(0, 32):
		for j in range(0, 32):
			a = train_lab[n][i][j][1]
			b = train_lab[n][i][j][2]		
			if a > max_a:
				max_a = a
			if a < min_a:
				min_a = a
			if b > max_b:
				max_b = b
			if b < min_b:
				min_b = b
'''
min_a: -86
max_a: 98
min_b: -107
max_b: 94
'''