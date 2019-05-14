import numpy as np
from pathlib import Path
from skimage import color
from scipy.ndimage import gaussian_filter
from extract_cifar10 import *
from data_utils import *
from matplotlib.pyplot import imread

'''
path = str(Path().absolute().parent)
path += '/cifar-10-python/cifar-10-batches-py/'
no_of_batches = 5
train_rgb, train_labels, qmat_rgb, qmat_labels = load_cifar10_data(path, no_of_batches)


# Convert data from RGB to LAB
N = len(train_rgb)
train_lab = np.ndarray(shape=(N, 32, 32, 3), dtype=np.int8, order='C') #HARA: THIS SEEMS TO BE DUPLICATE OF LINE 32 IN extract_cifar10.py
for n in range(0, N):
	if n%1000==0:
		print(n,' images converted to LAB.')	
	train_lab[n] = color.rgb2lab(train_rgb[n])
np.save('train_lab', train_lab)
'''
'''
yellow = imread('yellow.png')
blue = imread('blue.png')
pink = imread('pink.png')
colours = np.array([yellow, pink, blue])
for n in range(0, 3):
	colours[n] = color.rgb2lab(colours[n])
	print('a: ', colours[n][0][0][1])
	print('b: ', colours[n][0][0][2])
'''
'''
grid_size = 10
inf_a = -95
sup_a = 105
bins_a = int((sup_a-inf_a)/grid_size)
inf_b = -115
sup_b = 95
bins_b = int((sup_b-inf_b)/grid_size)
#train_lab = np.load('train_lab.npy')
#qmat_test = imgs2qmat(colours, grid_size, inf_a, sup_a, inf_b, sup_b)

qmat1 = imgs2qmat(train_lab[0:10000], grid_size, inf_a, sup_a, inf_b, sup_b)
qmat2 = imgs2qmat(train_lab[10000:20000], grid_size, inf_a, sup_a, inf_b, sup_b)
qmat3 = imgs2qmat(train_lab[20000:30000], grid_size, inf_a, sup_a, inf_b, sup_b)
qmat4 = imgs2qmat(train_lab[30000:40000], grid_size, inf_a, sup_a, inf_b, sup_b)
qmat5 = imgs2qmat(train_lab[40000:50000], grid_size, inf_a, sup_a, inf_b, sup_b)
'''
#qmat = qmat1+qmat2+qmat3+qmat4+qmat5
qmat = np.load('qmat.npy')
# apply Gaussian filter to qmat before saving
qmat_gaussian = gaussian_filter(qmat, sigma=5)
sign_mat = np.sign(qmat).astype(np.int)
qmat_smooth = np.multiply(sign_mat, qmat_gaussian)
#print(sign_mat,'\n')
Q = int(np.sum(sign_mat))
#print(sign_mat)
cifar_pts_in_hull = np.zeros((Q,2), dtype=np.int64)
n = 0
a = -90
for i in range(0, sign_mat.shape[0]):
	b = -110
	for j in range(0, sign_mat.shape[1]):
		if sign_mat[i][j] == 1:
			cifar_pts_in_hull[n][0] = a
			cifar_pts_in_hull[n][1] = b
			n += 1
		b += 10
	a += 10

#desired_pts = np.load('pts_in_hull_org.npy')
#np.save('cifar_pts_in_hull', cifar_pts_in_hull)

#desired_prior = np.load('prior_probs_org.npy')
tot = np.sum(qmat_smooth)
prob = qmat_smooth/tot
print(np.max(prob))
q = 0

prior_probs = np.ndarray(Q, dtype=np.float64)
for i in range(0, sign_mat.shape[0]):
	for j in range(0, sign_mat.shape[1]):
		elem = prob[i][j]
		if elem != 0:
			prior_probs[q] =  elem
			q += 1

#print(prior_probs)
np.save('cifar_prior_probs', prior_probs)