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


def save_encoder(model):
    with open(data_utils.get_model_name_from_propeties(True, params['layers_sizes']), 'wb') as output:
        w = WeightMatrices(model.trained_matrices,model.trained_biases);
        pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
    
    del w

def q(x):
        if(np.ndim(x) == 0):
            print(1);
        elif(np.ndim(x) == 1):
            print(len(x))
        else: print((np.size(x,0),np.size(x,1)));


    
    


mnist_provider = MNISTDataProvider();

if(False): #Rip RBM 
    
    params = {
      'epochs': 2,
      'learning_rate':0.01,
      'batch_size': 100,
      'free_energy_batch_size': 500,

      'validate': True,
      'shuffle': False,
      'gibbs_sampling_steps': 1,
      'layers_qtty': 1,
      # [n_input_features, layer_1, ...]
      'layers_sizes': [784,100],
      'early_stopping_number':5

      }
    if(True): #Train
    
    
        notes = 'train_layers_by_pairs__'
        ModelClass = RBMTrainByPairs
     
        params['notes'] = notes
        initial_params = dict(params)
        
        
        mnist_provider = MNISTDataProvider();
        test_run_no = None
        
        rbm_model = ModelClass(
            data_provider=mnist_provider,
            params=params)
        summary = [];
        rbm_model.train(summary)
      

        #print(rbm_model.__dict__);
       # biases.append(getattr(rbm_model, "bias_0"));
        #for layer in range(0, params['layers_qtty']):
        #    matrices.append(getattr(rbm_model, "W_{a}_{b}".format(a = layer, b = layer+1)).eval(rbm_model.sess));
        #    biases.append(getattr(rbm_model, "bias_{a}".format(a = layer+1)).eval(rbm_model.sess));
        #    print("--")

        
        with open(data_utils.get_model_name_from_propeties(False, params['layers_sizes']), 'wb') as output:
            w = WeightMatrices(rbm_model.trained_matrices,rbm_model.trained_biases);
            pickle.dump(w, output, pickle.HIGHEST_PROTOCOL)
        
        del w
        
        with open(data_utils.get_model_name_from_propeties(False, params['layers_sizes']), 'rb') as input:
            w = pickle.load(input)

            print(w.matrices);
            print(w.biases);

        #for layer in range(0, params['layers_qtty']):
            
        #    print(getattr(rbm_model, 'W_{a}_{b}'.format(a=layer, b=layer+1)));
    if(False): #Test
        ModelClass = RBMTrainByPairs
        rbm_model = ModelClass(
            params = params,
            data_provider=mnist_provider
           )
        
        rbm_model.test(
            run_no=0,
            plot_images=mnist_provider.test,
        )
if(True):#Encoder
    params = {
        'epochs': 300,
        'learning_rate': 100,
        'batch_size': 100,
        'ridge':0.,
        'validate': True,
        'shuffle': True,
        'layers_qtty': 2,
        # [n_input_features, layer_1, ...]
        'layers_sizes': [784,150],
        'without_noise': True,
        'early_stopping_number':10
    }
    model = EncoderSimple(
    data_provider=mnist_provider,
    params=params,
    rbm_run_no=0)
    if(True):#Train
        model.train()
        save_encoder(model);
        

          
    if(True): #Test
        model.test(
            run_no=0,
            plot_images=mnist_provider.test,
        )
if(False):#Keras encoder
    params = {
       'epochs': 100,
       'learning_rate': 100,
       'batch_size': 100,
       'ridge':0.,
       'validate': True,
       'shuffle': True,
       'layers_qtty': 1,
       # [n_input_features, layer_1, ...]
       'layers_sizes': [784,100],
       'without_noise': True,
       'early_stopping_number':10
    }
    layer_sizes_to_use = params['layers_sizes'];
    for l in reversed( params['layers_sizes'][:-1]):
        layer_sizes_to_use.append(l);
    print(layer_sizes_to_use)
    def build_model(layers_sizes, layer_is_fixed):
            model = keras.Sequential();
            for i in range(0, len(layers_sizes)-1):
                model.add(keras.layers.Dense(units=layers_sizes[i+1], input_shape =(layers_sizes[i],),bias_initializer='random_uniform', kernel_initializer='random_uniform', activation=tf.nn.sigmoid))
   
            #
            model.compile(optimizer = tf.train.GradientDescentOptimizer(0.01),
              loss='binary_crossentropy',
              metrics=['acc'])
            #model.compile(loss='mse',
            #           optimizer=optimizer,
            #           metrics=['mae'])
            return model
    layer_is_fixed = [];
    for i in range(0, len(layer_sizes_to_use)):
        layer_is_fixed.append(False);
    model = build_model(layer_sizes_to_use,layer_is_fixed);
    
    
    EPOCHS = 100
    
    #Store training stats
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
    
    
    history = model.fit(mnist_provider.train.train,mnist_provider.train.train, epochs=EPOCHS,
                       validation_split=0.2, verbose=1,shuffle=False,
                       callbacks=[early_stop]) #PrintDot()
    
    
    [loss, mae] = model.evaluate(mnist_provider.test.train,mnist_provider.test.train, verbose=0)
    print("Acc "  + str(round(history.history['acc'][-1],4)) +" Stopped at: " + str(len(history.history['val_loss'])) );

    
   # rbm_model = ModelClass(
   #     data_provider=mnist_provider,
   #     params=params)





