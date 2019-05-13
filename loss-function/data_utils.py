import numpy as np
from sklearn.neighbors import NearestNeighbors

# this function returns ranges of ab-values, not ranges of bins
def get_ab_ranges(train_lab):
	min_a = 127
	min_b = 127
	max_a = -128
	max_b = -128
	for n in range(0, len(train_lab)):
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
	return min_a, min_b, max_a, max_b

# bins are inclusive from below but not from above, e.g. -5 to 4, or 15 to 24
def imgs2qmat(train_lab, grid_size, inf_a, sup_a, inf_b, sup_b):
	bins_a = int((sup_a-inf_a)/grid_size)
	bins_b = int((sup_b-inf_b)/grid_size)
	qmat = np.zeros((bins_a, bins_b))
	
	train_lab = train_lab[:,:,:,1:3]
	inf_array = np.array([inf_a,inf_b])
	train_lab = train_lab - inf_array
	train_lab = train_lab / grid_size
	
	for n in range(0, len(train_lab)):
		if n%1000==0:
			print(n,' images converted to matrix.')	
		for i in range(0, 32):
			for j in range(0, 32):
				k = np.floor(train_lab[n][i][j][0])
				l = np.floor(train_lab[n][i][j][1])
				qmat[k][l] += 1
			
	return qmat, train_lab

# lab_bins stores for each pixel the bin (starting at 0, up to Q) corresponding to that pixel
def get_lab_bins(qmat, bin_indices):
	Q = int(np.sum(np.sign(qmat)))
	print('Q: ', Q)
	bins_splited = np.split(bin_indices,1000)
	for setnumber subset in enumerate(bins_splited):
		lab_bins = np.memmap('QVectors_' + setnumber, mode='w+', shape = (len(subset),32,32,Q), dtype ='float64')
		for n in range(0,len(subset)):
			for i in range(0,32):
				for j in range(0,32):
					bin_idx = subset[n][i][j]			
					Qvector = get_bin(qmat, bin_idx)
					lab_bins[n][i][j] = Qvector
	return

def get_bin(qmat, bin_idx):
	k = bin_idx[0]
	l = bin_idx[1]
	
	# sign_mat is used to tell whether a particular bin exists or not
	sign_mat = np.sign(qmat)
	
	Q = int(np.sum(sign_mat)) #how many bins
	todefine = 0 #index of the next bin that needs to be defined
	
	bin_locations = np.array(Q)
	for i in range(0,len(sign_mat)):
		for j in range(0,len(sign_mat[0])):
			if(sign_mat[i][j] == 1):
				bin_locations[todefine] = (i,j)
				
	neigh = NearestNeighbors(n_neighbors=5)
	neigh.fit(bin_locations)
	neigh.kneighbors([(k,l)])
	'''
	k = np.floor(kfloat)
	l = np.floor(lfloat)
	kfloat = bin_idx[0] - k
	lfloat = bin_idx[1] - l
	
	Qarray = np.zeros(qmat.shape)
	Qarray[k][l] = 36/256 #The bin the pixel sits on, it will always exist, so no need to check sign_mat
	bins_found = 1
	
	if (kfloat =< 0.5):
		kside = -1
	else:
		kside = 1
		
	if (lfloat =< 0.5):
		lside = -1
	else:
		lside = 1
		
	if (sign_mat[k+kside][l] == 1):
		Qarray[k+kside][l] = 24/256
		bins_found++
	if (sign_mat[k][l+lside] == 1):
		Qarray[k][l+lside] = 24/256
		bins_found++
	if (sign_mat[k+kside][l+lside] == 1):
		Qarray[k+kside][l+lside] = 16/256
		bins_found++
	
	if (abs(kfloat-0.5) <= abs(lfloat-0.5))
		
	if (sign_mat[k-kside][l] == 1):
		Qarray[k-kside][l] = 24/256
		bins_found++
	else:
	if (sign_mat[k][l-lside] == 1):
		Qarray[k][l-lside] = 24/256
		bins_found++
	'''	
	
	
	
	
	
	
	
	sign_flat = sign_mat.flatten()
	indices = sign_flat.nonzero()
	bin_val = np.where(indices[0]==bin_idx)[0]
	
	return Qvector
