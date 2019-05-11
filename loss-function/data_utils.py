import numpy as np

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
	bin_indices = np.ndarray(shape=(len(train_lab), 32, 32, 1), dtype=np.uint16, order='C') # boolean dtype?
	for n in range(0, len(train_lab)):
		if n%1000==0:
			print(n,' images converted to matrix.')
			
		for i in range(0, 32):
			for j in range(0, 32):
				a = train_lab[n][i][j][1]
				b = train_lab[n][i][j][2]
				k = 0;
				l = 0;
				while k < bins_a:
					if a < (inf_a + (1+k)*grid_size):
						while l < bins_b:
							if b < (inf_b + (1+l)*grid_size):
								bin_idx = int(k*bins_b + l)
								bin_indices[n][i][j][0] = bin_idx
								qmat[k][l] += 1
								break
							else:
								l += 1
						break
					else:
						k += 1
			
	return qmat, bin_indices

# lab_bins stores for each pixel the bin (starting at 0, up to Q) corresponding to that pixel
def get_lab_bins(qmat, bin_indices):
	Q = int(np.sum(np.sign(qmat)))
	print('Q: ', Q)
	lab_bins = np.zeros((50000,32,32,1), dtype=np.int16)
	for n in range(0,len(bin_indices)):
		for i in range(0,32):
			for j in range(0,32):
				bin_idx = bin_indices[n][i][j][0] 			
				bin_val = get_bin(qmat, bin_idx)
				lab_bins[n][i][j] = bin_val

	return lab_bins

def get_bin(qmat, bin_idx):
	sign_mat = np.sign(qmat)
	Q = int(np.sum(sign_mat))
	sign_flat = sign_mat.flatten()
	indices = sign_flat.nonzero()
	bin_val = np.where(indices[0]==bin_idx)[0]
	return bin_val
'''
a = np.array([[0,1,1],[1,0,1],[0,0,1]])
print(a)
b = 8
print(get_bin(a,b))
'''
