import numpy as np
from tensorflow.examples.tutorials import mnist
from src.lab_4 import data_utils


class TrainTest:
    def __init__(self,data,target):
        self.data = data;
        self.target  = target;

class BaseDataProvider:
    def get_shapes(self):
        """Return shapes of inputs and targets"""
        raise NotImplementedError

    def get_train_set(self):
        raise NotImplementedError

    def get_validation_set(self):
        raise NotImplementedError

    def get_test_set(self):
        raise NotImplementedError

    def get_train_set_iter(self, batch_size):
        """Return generator with train set"""
        raise NotImplementedError

    def get_validation_set_iter(self, batch_size):
        """Return generator with validation set"""
        raise NotImplementedError

    def get_test_set_iter(self, batch_size):
        """Return generator with test set"""
        raise NotImplementedError


class MNISTDataProvider(BaseDataProvider):
    def __init__(self, bin_code_width=100,validation_size=0.02):
        train_data_bin =  data_utils.matrix_from_file("bindigit_trn", "csv");
        test_data_bin =  data_utils.matrix_from_file("bindigit_tst", "csv");
        train_data_target =  data_utils.matrix_from_file("targetdigit_trn", "csv");
        test_data_target =  data_utils.matrix_from_file("targetdigit_tst", "csv");
        # split validation at 99%
        
        validation_split_rate = validation_size;
        validation_split_index = int(round(np.size(train_data_bin, 0)- np.size(train_data_bin, 0)*validation_split_rate));
        self._shapes = {};
        
        (train_data_bin_sorted, train_data_target_sorted) = data_utils.sort_data_set(train_data_bin[:validation_split_index,],  train_data_target[:validation_split_index,]);
        self.train = TrainTest(train_data_bin_sorted,train_data_target_sorted);
        #print(train_data_bin[:validation_split_index,].shape)
        #self.train = TrainTest(train_data_bin[:validation_split_index,],  train_data_target[:validation_split_index,]);
        self.validation = TrainTest(train_data_bin[validation_split_index:,], train_data_target[validation_split_index:,]);

        

        self._shapes["inputs"] = np.size(train_data_bin, 1);
        self._shapes["targets"] = np.size(train_data_target, 1);
        
        
        self.test = TrainTest(test_data_bin, test_data_target);

        self.bin_code_width = bin_code_width;
        
        

    @property
    def shapes(self):
        return self._shapes


    def get_train_set(self):
        data = self.mnist.data
        return data.images, data.labels

    def get_validation_set(self):
        data = self.mnist.validation
        return data.images, data.labels

    #def get_test_set(self):
      #  data = self.mnist.test
       # return data.images, data.labels;
        

    def get_generator(self, data, batch_size, shuffle=True, noise_data=None):
        quantity = np.size(data.target,0);
        #if noise_data is None:
        #    noise_perm = None
        #if shuffle:
        #    indexes = np.random.permutation(quantity)
        #    images_perm = data.images[indexes]
        #   labels_perm = data.labels[indexes]
        #    if noise_data is not None:
        #        noise_perm = noise_data[indexes]
        #else:
        #    images_perm = data.images
        #    labels_perm = data.labels
        #    noise_perm = noise_data
        for i in range(quantity // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size
            images = data.data[start: end,:]
            labels = data.target[start: end,:]
            #if noise_perm is not None:
            #    noise = noise_perm[start: end,:]
            #    yield images, labels, noise
            #else:
            yield images, labels

    def get_train_set_iter(self, batch_size, shuffle=True, noise=False):
        data = self.train;
       # noise_data = None
       # if noise:
        #    noise_data = self.noise_train
       # return self.get_generator(data, batch_size, shuffle, noise_data)
        return self.get_generator(data, batch_size, shuffle, data)


    def get_validation_set_iter(self, batch_size, shuffle=True, noise=False):
        data = self.validation;
        #print(data.data.shape)
        while(True):
            yield data.data,data.target;#data.data[:batch_size,], data.target[:batch_size,]

        #return self.get_generator(data, batch_size, shuffle)

    def get_test_set_iter(self, batch_size, shuffle=True, noise=False):
        data = self.validation;
        return self.get_generator(data, batch_size, shuffle)

    
    
    
    
    
    
    
    