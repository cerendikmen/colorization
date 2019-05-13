import numpy as np
import matplotlib.pyplot as plt
import pickle


def unpickle(file):
	'''Load byte data from file'''
	with open(file, 'rb') as f:
		data = pickle.load(f, encoding='latin-1')
	return data


def load_cifar10_data(data_dir, no_of_batches): #HARA's EDIT PENDING REVIEW
	'''Return train_data, train_labels, test_data, test_labels
	The shape of data is N x 32 x 32 x 3''' #HARA'S EDIT PENDING REVIEW
	train_data = None
	train_labels = []

	for i in range(1, 1+no_of_batches): #HARA's EDIT PENDING REVIEW
		data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
		if i == 1:
			train_data = data_dic['data']
		else:
			train_data = np.vstack((train_data, data_dic['data']))

		train_labels += data_dic['labels']

	test_data_dic = unpickle(data_dir + "/test_batch")
	test_data = test_data_dic['data']
	test_labels = test_data_dic['labels']

	train_data = train_data.reshape((len(train_data), 3, 32, 32))
	train_data = np.rollaxis(train_data, 1, 4)
	train_labels = np.array(train_labels)

	test_data = test_data.reshape((len(test_data), 3, 32, 32))
	test_data = np.rollaxis(test_data, 1, 4)
	test_labels = np.array(test_labels)

	return train_data, train_labels, test_data, test_labels