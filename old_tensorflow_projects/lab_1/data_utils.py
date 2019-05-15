'''
Created on 5 Sep 2018

@author: marcuspousette
'''
import numpy as np;
import matplotlib.pyplot as plt
import random;
def getTestBinaryTrainTestSet(linearly_seperable, offset=[0,0]):

    n_training = 200;
    n_test = 200;
    n_avg = int(round((n_training + n_test)/2));
    mu1 = [];
    sigma1 =[];
    mu2 = [];
    sigma2 = [];
    r1 = [];
    if(linearly_seperable):
        mu1 = np.add([1,0.5], offset);
       
        var1 = 0.25;
        var2 = 0.25;
        sigma1 = np.matrix([[var1, 0], [0, var1]]);
        mu2 =np.add([-0.5,0], offset);
        sigma2 = np.matrix([[var2, 0], [0, var2]]);
        r1 = np.random.multivariate_normal(mu1,sigma1,n_avg);
    else:
        mu1a = np.add([1,0.3], offset);
        mu1b = np.add([-1,0.3], offset);
        var1 = 0.2**2;
        var2 = 0.3**2;
        sigma1 = np.matrix([[var1, 0], [0, var1]]);
        mu2 = [0,-0.1];
        sigma2 = np.matrix([[var2, 0], [0, var2]]);
        r1 = None;
        for i in range(0, n_avg):
            if(r1 is not None):
                if(np.mod(i,2)== 0):
                    r1 = np.vstack((r1,np.random.multivariate_normal(mu1a,sigma1,1))); 
                else:
                    r1 = np.vstack((r1,np.random.multivariate_normal(mu1b,sigma1,1)));
             
            else:
                r1 = np.random.multivariate_normal(mu1a,sigma1,1); 
        

    r2 = np.random.multivariate_normal(mu2,sigma2,n_avg);
    # Classifiers
  
    ones_vector = np.ones(n_avg).reshape(n_avg,1);
    
    r1 = np.concatenate((r1, ones_vector), axis=1);  
    r2 = np.concatenate((r2, -ones_vector), axis=1);
    half_training = int(n_training/2)
    a_train = r1[0:half_training,:];
    a_test = r1[half_training:,:];
    b_train = r2[0:half_training,:];
    b_test = r2[half_training:,:];

    return BinaryTrainTestSet(a_train, a_test, b_train,b_test);
def get_wanted_set_permutations(train_test_set, use_rest_as_test_set = False):
    all_indices = range(0, np.size(train_test_set.class_a_set_training,0));
    samples = random.sample(all_indices, int(len(all_indices)*0.25));
    illegal_indices = set();
    for sample in samples:
        illegal_indices.add(sample);
    
    new_a = [];
    new_a_val = [];
    new_b = [];
    new_b_val = [];

    for i in all_indices:
        if(i not in illegal_indices):
            new_a.append(train_test_set.class_a_set_training[i,::+1])
            new_b.append(train_test_set.class_b_set_training[i,::+1])
        else:
            new_a_val.append(train_test_set.class_a_set_training[i,::+1])
            new_b_val.append(train_test_set.class_b_set_training[i,::+1])
    
    perm_1 =None;
    if (use_rest_as_test_set):
        perm_1 = BinaryTrainTestSet(np.vstack(new_a), np.vstack(new_a_val),np.vstack(new_b),  np.vstack(new_b_val));
    else:
        perm_1 = BinaryTrainTestSet(np.vstack(new_a), train_test_set.class_a_set_test,np.vstack(new_b), train_test_set.class_b_set_test);

   

    #50 % from class A
    samples = random.sample(all_indices, int(len(all_indices)*0.5));
    illegal_indices = set();
    for sample in samples:
        illegal_indices.add(sample);
        
    new_a = [];
    new_a_val = [];
    new_b = [];
    new_b_val = [];
    for i in all_indices:
        if(i not in illegal_indices):
            new_a.append(train_test_set.class_a_set_training[i,::+1])
            new_b.append(train_test_set.class_b_set_training[i,::+1])
        else:
            new_a_val.append(train_test_set.class_a_set_training[i,::+1])
            new_b_val.append(train_test_set.class_b_set_training[i,::+1])


    perm_2 = None;
    perm_3 = None;
    if(use_rest_as_test_set):
        perm_2 = BinaryTrainTestSet(np.vstack(new_a), np.vstack(new_a_val), train_test_set.class_b_set_training, np.array([], dtype=np.int64).reshape(0,3));
        perm_3 = BinaryTrainTestSet(train_test_set.class_a_set_training, np.array([], dtype=np.int64).reshape(0,3), np.vstack(new_b),  np.vstack(new_b_val));
    else:
        perm_2 = BinaryTrainTestSet(np.vstack(new_a), train_test_set.class_a_set_test,train_test_set.class_b_set_training, train_test_set.class_b_set_test);
        perm_3 = BinaryTrainTestSet(train_test_set.class_a_set_training, train_test_set.class_a_set_test,np.vstack(new_b), train_test_set.class_b_set_test);
    
    # X coordinate 20 and 80% pick
    # x < 0 
    train_set_less = [];
    train_set_above = [];
    for i in range(0,np.size( train_test_set.class_a_set_training,0)):
        if(train_test_set.class_a_set_training[i,0] < 0):
            train_set_less.append(train_test_set.class_a_set_training[i,::+1])
        else:
            train_set_above.append(train_test_set.class_a_set_training[i,::+1])
            
    train_set_above = np.vstack(train_set_above);
    train_set_less = np.vstack(train_set_less);
    all_indices_less = range(0, np.size(train_set_less,0));
    all_indices_above = range(0, np.size(train_set_above,0));

    samples_20 = random.sample(all_indices_less, int(np.round(np.size(train_set_less,0)*0.2)));
    samples_80 = random.sample(all_indices_above, int(np.round(np.size(train_set_above,0)*0.8)));
    illegal_indices_20 = set();
    illegal_indices_80 = set();

    for sample in samples_20:
        illegal_indices_20.add(sample);
    for sample in samples_80:
        illegal_indices_80.add(sample);
        
    new_a = [];
    new_a_val = [];

    for i in range(0, np.size(train_set_less,0)):
        if(i not in illegal_indices_20):
            new_a.append(train_set_less[i,::+1])
        else:
            new_a_val.append(train_set_less[i,::+1])

    for i in range(0, np.size(train_set_above,0)):
        if(i not in illegal_indices_80):
            new_a.append(train_set_above[i,::+1])
        else:
            new_a_val.append(train_set_above[i,::+1])

    perm4 = None;
    if(use_rest_as_test_set):
        perm_4 = BinaryTrainTestSet(np.vstack(new_a), np.vstack(new_a_val),train_test_set.class_b_set_training, np.array([], dtype=np.int64).reshape(0,3));
    else:
        perm_4 = BinaryTrainTestSet(np.vstack(new_a), train_test_set.class_a_set_test,train_test_set.class_b_set_training, train_test_set.class_b_set_test);

    set_of_sets = {};
    set_of_sets['Default'] = train_test_set;
    set_of_sets['Remove 25%'] = perm_1
    set_of_sets['Remove 50% of red'] =  perm_2;
    set_of_sets['Remove 50% of blue'] = perm_3;
    set_of_sets['Remove 20% of red less than 0 and 80% above'] = perm_4;
    return set_of_sets;
class BinaryTrainTestSet:
    def __init__(self, class_a_set_training,class_a_set_test, class_b_set_training,class_b_set_test ):
            self.class_a_set_training = class_a_set_training;
            self.class_a_set_test = class_a_set_test;
            self.class_b_set_training = class_b_set_training;
            self.class_b_set_test = class_b_set_test;
            # Data to use
            train = np.concatenate((class_a_set_training, class_b_set_training));
            test = np.concatenate((class_a_set_test, class_b_set_test));

            # Shuffles the rows
            np.random.shuffle(train);    
            np.random.shuffle(test);    

         
            self.train_set = ObservationTargetBinary(train)
            self.test_set = ObservationTargetBinary(test)
            
    def plot_classes(self):
        r1 = self.class_a_set_training;
        r2 = self.class_b_set_training;  
        r3 = self.class_a_set_test;
        r4 = self.class_b_set_test;      
        plt.scatter(r1[:,0], r1[:,1],  c="r")
        plt.scatter(r2[:,0], r2[:,1],  c="b")
        plt.scatter(r3[:,0], r3[:,1],  c="r",marker=">")
        plt.scatter(r4[:,0], r4[:,1],  c="b",marker=">")
        plt.show();
        
class ObservationTargetBinary:
    def __init__(self, s):
        self.observation = s[:,:np.size(s,1) -1];
        self.target = s[:,np.size(s,1)-1];

class ObservationTarget:
    def __init__(self, pattern, target):
        self.observation =pattern;
        self.target = target;
