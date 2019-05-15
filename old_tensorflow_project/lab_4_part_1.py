import numpy as np;
import matplotlib.pyplot as plt;
import sys;
from src.lab_4 import data_utils
import os
import argparse
import numpy as np
import tensorflow as tf;
from src.lab_4.rbm_train_by_pairs import RBMTrainByPairs
from src.lab_4.data_providers import MNISTDataProvider
from src.lab_4.autoencoder import Encoder;
import pickle;
from src.lab_4.weight_matrices import WeightMatrices
from src.lab_4.autoencoder_simple import EncoderSimple
import tensorflow.keras as keras;
from tensorflow.keras.utils import to_categorical;

#https://ikhlestov.github.io/posts/rbm-based-autoencoders-with-tensorflow/


def save_encoder(model, train_summary):
    with open(data_utils.get_model_name_from_propeties(True, params['layers_sizes']), 'wb') as output:
        w = WeightMatrices(model.trained_matrices,model.trained_biases,train_summary);
        pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
    
    del w

def q(x):
        if(np.ndim(x) == 0):
            print(1);
        elif(np.ndim(x) == 1):
            print(len(x))
        else: print((np.size(x,0),np.size(x,1)));


    
    




        
if(False):#Encoder best ridge?
    mnist_provider = MNISTDataProvider(validation_size=0.02);

    # try three different ridge settings
    ridges =[0,0.0000001,0.000001,0.00001,0.0001]
    #end_sizes_2 = [8**2,9**2,10**2,11**2];
    #end_sizes_2 = [7**2,8**2,9**2,10**2];

    for ridge in ridges:
        params = {
            'epochs': 100,
            'learning_rate': 10,
            'batch_size': 100,
            'ridge':ridge,
            'validate': True,
            'verbose' :False,
            'shuffle': True,
            'layers_qtty': 1,
            # [n_input_features, layer_1, ...]
            'layers_sizes': [784,100],
            'without_noise': True,
            'early_stopping_number':10
        }
        summary = [];

        model = EncoderSimple(
        data_provider=mnist_provider,
        params=params)
        if(True):#Train
            model.train(summary)
            print("Last validation error for ridge: {a} is {b}".format(a=ridge,b=summary[-1][1]));
if(False):#Encoder 1 Hidden Layer
    mnist_provider = MNISTDataProvider();

    end_sizes = [50,65,80,95,110,120];
    converge_arr = [];
    for size in end_sizes:
        params = {
            'epochs': 300,
            'learning_rate': 10,
            'batch_size': 100,
            'ridge':0.00001,
            'validate': True,
            'shuffle': True,
            'layers_qtty': 1,
            'verbose':False,
            # [n_input_features, layer_1, ...]
            'layers_sizes': [784,size],
            'without_noise': True,
            'early_stopping_number':10,
            'run_no':0
        }
        encoder = EncoderSimple;
       
        model = encoder(
        data_provider=mnist_provider,
        params=params)
        
        if(True):#Train
            summary = [];
            model.train(summary)
            
            save_encoder(model,summary);   
            plt.plot(np.vstack(summary)[:,0],np.vstack(summary)[:,1], label="784 to " + str(size) + " network");
 
        
        if(True): #Test
            model.test(0, plot_images=mnist_provider.test.data);
            print("For layer sizes " + str(params['layers_sizes']) + " test error and sparseness: " + str(model.get_test_error(run_no=0 )));
    plt.legend();
    plt.ylabel("Reconstruction error rate");
    plt.xlabel("Epoch");
    plt.show();
    
    
if(True):#Encoder 1,2,3 hidden layers
    mnist_provider = MNISTDataProvider(validation_size=0.03);


    #layer_configuration = [784,121,81,49,25];
    layer_configuration = [784,80];
    converge_arr = [];
    for hl in range(0,len(layer_configuration)-1):
        print("CONF: " + str(layer_configuration[:(hl+2)]));
        params = {
            'epochs': 30,
            'learning_rate': 5,
            'batch_size': 10,
            'ridge':0.01,
            'validate': True,
            'shuffle': True,
            'layers_qtty': hl+1,
            'verbose':True,
            # [n_input_features, layer_1, ...]
            'layers_sizes': layer_configuration[:(hl+2)],
            'without_noise': True,
            'early_stopping_number':10,
            'run_no':0
        }
        encoder = EncoderSimple;
       
        model = encoder(
        data_provider=mnist_provider,
        params=params)
        
        if(True):#Train
            summary = [];
            model.train(summary)
            save_encoder(model,summary);   
           # plt.plot(np.vstack(summary)[:,0],np.vstack(summary)[:,1], label="Config.: " + str(params['layers_sizes']));
 
        
        if(True): #Test
           # model.test(0, plot_images=mnist_provider.test.data);
            model.test(0, plot_images=mnist_provider.test.data);
            print("For layer sizes " + str(params['layers_sizes']) + " test error and sparseness: " + str(model.get_test_error(run_no=0 )));
 
 
if(False):#Lasso ? 
    mnist_provider = MNISTDataProvider(validation_size=0.02);

    # try three different ridge settings
    #end_sizes_2 = [8**2,9**2,10**2,11**2];
    #end_sizes_2 = [7**2,8**2,9**2,10**2];

    params = {
        'epochs': 10,
        'learning_rate': 10,
        'batch_size': 10,
        'ridge':0.01,
        'lasso':0.,
        'validate': True,
        'verbose' :True,
        'shuffle': True,
        'layers_qtty': 1,
        # [n_input_features, layer_1, ...]
        'layers_sizes': [784,100],
        'without_noise': True,
        'early_stopping_number':10
    }
    summary = [];

    model = EncoderSimple(
    data_provider=mnist_provider,
    params=params)
    if(True):#Train
        model.train(summary)
        model.test(0, plot_images=mnist_provider.test.data);

       # print("Last validation error for ridge: {a} is {b}".format(a=ridge,b=summary[-1][1]));
    