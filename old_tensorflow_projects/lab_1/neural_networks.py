'''
Created on 5 Sep 2018

@author: marcuspousette
'''
import matplotlib.pyplot as plt
from src.lab_1 import math_utils
import numpy as np;
import random
from numpy import absolute

class ErrorEvaluatorMissClassifications:
    @staticmethod
    def  get_error( outputs,targets):
        return np.sum(np.abs(targets-np.sign(outputs))/2);    

class ErrorEvaluatorMissClassificationsArray:
    @staticmethod
    def get_error(outputs, targets):
        error = 0;
        for i in range(0, np.size(outputs,1)):
            error +=  min(np.sum(np.abs(targets.T[:,i]-np.sign(outputs)[:,i])/2),1);  
        return error;
class ErrorEvaluatorMeanSquare:
    @staticmethod
    def get_error(outputs, targets):
      
        return np.std((outputs-targets.T));
np.random.seed(3335);
class SingleLayerNetwork:
    
    def __init__(self, dimension_in, dimension_out, use_bias=True,W=None):
        
        
        if(use_bias is not None and use_bias):
            dimension_in = dimension_in +1;
        self.use_bias = use_bias;
        if(W is not None):
          
            self.W = W;
        else:
            self.W = np.random.rand(dimension_out,dimension_in);
            self.W = np.array((self.W - 0.5)*2); # mean and scale shift
        self.W_start = self.W;
   
    def train_binary(self, nepochs,eta, train_test_set, sequential_training=False, use_delta_rule=False, ev=ErrorEvaluatorMissClassifications.get_error, activation_function = np.sign):
        return self.train(nepochs,eta,  train_test_set.train_set, train_test_set.test_set, sequential_training, use_delta_rule)
    def train(self, nepochs,eta, train_set, test_set, sequential_training=False, use_delta_rule=False,ev=ErrorEvaluatorMissClassifications.get_error,activation_function = np.sign):
        number_of_observations_train = np.size(train_set.observation,0);
        number_of_observations_test = np.size(test_set.observation,0);
        

        ones_arr_train =np.reshape(np.ones(number_of_observations_train),[number_of_observations_train, 1]);
        ones_arr_test =np.reshape(np.ones(number_of_observations_test),[number_of_observations_test, 1]);
     
        X = [];
        X_test = [];

        if(self.use_bias):
            X = np.hstack((train_set.observation,ones_arr_train)).T;
            X_test = np.hstack((test_set.observation,ones_arr_test)).T;
        else:
            X = train_set.observation.T;
            X_test =test_set.observation.T;
            
    
        t = train_set.target;
        t_test = test_set.target;
        error_vector = ev(self.predict(X_test), t_test);#float(self.get_number_of_missclassifications(self.predict(X_test), t_test))/number_of_observations_test;  
       
        X_iter = [];
        t_iter = [];
       
        
        if(sequential_training):
            for i in range(0, np.size(X,1)):   
                X_iter.append(X[:,i:i+1]);
                t_iter.append(t[i]);
        else:
            X_iter.append(X);
            t_iter.append(t);
        if(use_delta_rule):
            for i in range(0, nepochs):
                for j in range(0, len(X_iter)):    
                    X_mat = X_iter[j];#np.reshape(X_iter[j],[np.size(X_iter[j],0),np.ndim(X_iter[i])]);
                    dW = -eta*np.dot((self.W.dot(X_mat) - t_iter[j]),X_mat.T);
                    self.W = self.W+dW;
                    if(j == len(X_iter) -1):
                        error = ev(self.predict(X_test), t_test);#self.get_number_of_missclassifications(self.predict(X_test), t_test);                    
                        error_vector = np.hstack((error_vector, float(error)/number_of_observations_test));
        else:
            for i in range(0, nepochs):
                for j in range(0, len(X_iter)):    
                    X_mat = X_iter[j];#np.reshape(X_iter[j],[np.size(X_iter[j],0),np.ndim(X_iter[i])]);
                    dW = -eta*np.dot((activation_function(self.W.dot(X_mat)) - t_iter[j]),X_mat.T);
                    self.W = self.W+dW;
                    if(j == len(X_iter) -1):
                        error = ev(self.predict(X_test), t_test);#self.get_number_of_missclassifications(self.predict(X_test), t_test);                    
                        error_vector = np.hstack((error_vector, float(error)/number_of_observations_test));


        #print(error_vector)
        #plt.show();
        
        return error_vector
        
        
    def predict(self, x):
        return np.sign(self.W.dot(x));
    
    def get_number_of_missclassifications(self, outputs, targets):
        return np.sum(np.abs(targets-outputs)/2);
    
class TwoLayerPerceptron:
    
    def __init__(self, dimension_in, number_of_hidden_nodes, dimension_out, use_bias=True,A=None,B=None):
        self.number_of_hidden_nodes_A = number_of_hidden_nodes;
        self.number_of_hidden_nodes_B = number_of_hidden_nodes;

        if(use_bias is not None and use_bias):
            dimension_in = dimension_in +1;
            self.number_of_hidden_nodes_A = number_of_hidden_nodes;
            self.number_of_hidden_nodes_B = number_of_hidden_nodes +1;
        self.use_bias = use_bias;

        if(A is not None and B is not None):
            self.A = A;
            self.B = B;      
        else:
            self.A = np.random.rand(self.number_of_hidden_nodes_A,dimension_in);
            self.A = np.array((self.A - 0.5)*2);
            
            
            self.B = np.random.rand(dimension_out,self.number_of_hidden_nodes_B);
            self.B = np.array((self.B - 0.5)*2);
        self.A_start = self.A;
        self.B_start = self.B;

        
    def train_binary(self, nepochs,eta,alpha, train_test_set, sequential_training=False, return_training_error=False):
        return self.train(ErrorEvaluatorMissClassifications,nepochs,eta,alpha,  train_test_set.train_set, train_test_set.test_set, sequential_training, return_training_error)
    def train(self,error_evaluator,nepochs,eta,alpha, train_set, test_set, sequential_training=False, return_training_error=False, absolute_errors=False):
        
        number_of_observations_train = np.size(train_set.observation,0);
        number_of_observations_test = np.size(test_set.observation,0);
        
      

        ones_arr_train =np.reshape(np.ones(number_of_observations_train),[number_of_observations_train, 1]);
        
        ones_arr_test =np.reshape(np.ones(number_of_observations_test),[number_of_observations_test, 1]);
        X = [];
        X_test = [];
        
        if(np.ndim(train_set.observation) == 1):
            train_set.observation = np.mat(train_set.observation).T;
        if(np.ndim(test_set.observation) == 1):
            test_set.observation = np.mat(test_set.observation).T;

        if(self.use_bias):
            X = np.hstack((train_set.observation,ones_arr_train)).T;
            X_test = np.hstack((test_set.observation,ones_arr_test)).T;
        else:
            X = train_set.observation.T;
            X_test = test_set.observation.T;
            
        t = train_set.target;
        t_test = test_set.target;
        
        error_test_div = number_of_observations_test;
        error_train_div = number_of_observations_test;
        if(absolute_errors):
            error_test_div = 1;
            error_train_div =1;
        
    
       
       
        error_vector =  error_evaluator.get_error(self.predict(X_test), t_test);
        
        error_vector_training =  error_evaluator.get_error(self.predict(X), t);

        X_iter = [];
        t_iter = [];
        
        if(sequential_training):
            for i in range(0, np.size(X,1)):   
                X_iter.append(X[:,i:i+1]);
                t_iter.append(t[i]);
        else:
            X_iter.append(X);
            t_iter.append(t);
        dB = 0;
        dA = 0;
        for i in range(0, nepochs):
            for j in range(0, len(X_iter)):
               
                # Chain iteration
                X_mat = X_iter[j];#
            
                # Forward pass
                a_out_star = self.A.dot(X_mat);
               
                
                if(self.use_bias):
                    a_out_star = np.vstack((a_out_star,np.ones([1,np.size(a_out_star,1)])));
                a_out =  math_utils.phi(a_out_star);
                
               
             
                b_out_star = self.B.dot(a_out);
                b_out =  math_utils.phi(b_out_star);

                #Backward iter
                
                delta_b = np.multiply((b_out -t_iter[j]),math_utils.phi_derivative(b_out_star));
                
                dB = dB*alpha - (np.dot(delta_b,a_out.T))*(1 - alpha);

                self.B = self.B + dB*eta;
              
                delta_a = np.multiply(np.dot(self.B.T,delta_b),math_utils.phi_derivative(a_out_star))
              #  print("We want:");
              #  q(self.A);
                #print("We have:");
              #  q(np.dot(delta_a,X_mat.T));
                if(self.use_bias):
                    delta_a = delta_a[:-1];
                 
                dA = dA*alpha - (np.dot(delta_a,X_mat.T))*(1 - alpha);  
                self.A = self.A + dA*eta;
                error = error_evaluator.get_error(self.predict(X_test), t_test);
                #std_error = np.var(self.predict(X_test)- t_test);
                if(j == len(X_iter) -1):
                    if(return_training_error):
                        error_vector_training = np.hstack((error_vector_training, float(error_evaluator.get_error(self.predict(X_mat),  t_iter[j]))));
                    error_vector = np.hstack((error_vector, float(error)));

        
        #plt.show();
        if(not return_training_error):
            return error_vector
        else: return (error_vector,error_vector_training);
  
        
    def predict(self, x_in, has_bias = True, features_are_columns=False): # features as rows  
        x = x_in; 
        if(features_are_columns):
            x = x.T;
        if(np.ndim(x) == 1):
            x = np.mat(x);
        if(has_bias == False):
            x = np.vstack((x, np.ones([1,np.size(x,1)])))
        
        a_out_star = self.A.dot(x);
        if(self.use_bias):
            a_out_star = np.vstack((a_out_star,np.ones([1,np.size(a_out_star,1)])));
        a_out =  math_utils.phi(a_out_star);
        
      
        b_out_star = self.B.dot(a_out);
        b_out =  math_utils.phi(b_out_star);
        return b_out;
        
    
   # def get_number_of_missclassifications(self, outputs, targets):
   #     return np.sum(np.abs(targets-np.sign(outputs))/2);    
    def get_number_of_missclassifications(self, outputs, targets):
        error = 0;
        for i in range(0, np.size(outputs,1)):
            error +=  min(np.sum(np.abs(targets.T[:,i]-np.sign(outputs)[:,i])/2),1);  
        return error;
    
            
def q(x):
    if(np.ndim(x) == 0):
        print(1);
    elif(np.ndim(x) == 1):
        print(len(x))
    else: print((np.size(x,0),np.size(x,1)));


    