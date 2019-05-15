'''
Created on 5 Sep 2018

@author: marcuspousette
'''
import numpy as np;

def phi(x):
    return 2/(1+np.exp(-x)) -1;
def phi_derivative(x):
    phix = phi(x);    
    return np.multiply((1+phix),(1-phix))/2;



def findYFromNullSpace(W,x):
    x = np.reshape(x, [2,1]);
    x = np.hstack((x, np.reshape(np.ones(2), [2,1])));
    # Assum W = 1x3, W mu x = 0 2 dimensions,
    return x.dot(np.array([-W[0,0]/W[0,1],-W[0,2]/W[0,1]]))
    
def getSensitivtyAndSpecificity(nn, test_set):
        number_of_observations = np.size(test_set.observation,0);
        ones_arr =np.reshape(np.ones(number_of_observations),[number_of_observations, 1]);
  
        X_test = [];
        if(nn.use_bias):
            X_test = np.hstack((test_set.observation,ones_arr)).T;
        else:
            X_test = test_set.observation.T;
        
        t_test = test_set.target;
        t_predicted = nn.predict(X_test)[0];

        TT = 0.;
        T = 0.;
        NN = 0.;
        N = 0.;
        for i in range(0, len(t_test)):
            pred = np.sign(t_predicted[i]);
            if(t_test[i] == 1):
                if(t_test[i] == pred ):
                    TT += 1;
                T += 1;
          
            if(t_test[i] == -1):
                if(t_test[i] == pred):
                    NN += 1;
                N += 1;

    
        return (round(TT/T,2), round(NN/N,2));
def fun_func(x,y):
    return np.exp((-np.multiply(x,x)*0.1 -np.multiply(y,y)*0.1)) -0.5;

    