import tensorflow as tf;
from src.lab_4.rbm_base import  RBMBase
from src.lab_4.weight_matrices import WeightMatrices
from tensorflow import keras;
import pickle;
from src.lab_4 import data_utils, data_providers
import numpy as np;
from src.lab_4.data_providers import MNISTDataProvider
from tensorflow.keras.utils import to_categorical;
from tensorflow.keras import optimizers

# load weight matrices
data_providers = MNISTDataProvider();
   
def build_model(weight_matrices, train_all):
    model = keras.Sequential();
    for i in range(0, len(weight_matrices.matrices)):
        model.add(keras.layers.Dense(units=np.size(weight_matrices.matrices[i],1), input_shape =( np.size(weight_matrices.matrices[i],0),),bias_initializer=keras.initializers.Constant(value=weight_matrices.biases[i]), kernel_initializer=keras.initializers.Constant(value=weight_matrices.matrices[i]), activation=tf.nn.sigmoid, trainable=train_all))
        #model.add(keras.layers.Dense(units=np.size(weight_matrices.matrices[i],1), input_shape =( np.size(weight_matrices.matrices[i],0),),bias_initializer='random_uniform', kernel_initializer='random_uniform', activation=tf.nn.sigmoid, trainable=train_all))

    
    model.add(keras.layers.Dense(units=10, input_shape =( np.size(weight_matrices.matrices[-1],0),),kernel_initializer='random_uniform',activation='softmax'))
    #optimizer = tf.train.RMSPropOptimizer(0.01)
    
    model.compile(optimizer="adam",
      loss='categorical_crossentropy',
      metrics=['acc'])
    #model.compile(loss='mse',
    #           optimizer=optimizer,
    #           metrics=['mae'])
    return model
def build_model_normal(layer_sizes):
    model = keras.Sequential();
    for i in range(0, len(layer_sizes)-1):
        model.add(keras.layers.Dense(units=layer_sizes[i+1], input_shape =(layer_sizes[i],),activation=tf.nn.sigmoid))
        #model.add(keras.layers.Dense(units=np.size(weight_matrices.matrices[i],1), input_shape =( np.size(weight_matrices.matrices[i],0),),bias_initializer='random_uniform', kernel_initializer='random_uniform', activation=tf.nn.sigmoid, trainable=train_all))

    
    model.add(keras.layers.Dense(units=10, input_shape =( layer_sizes[-1],),kernel_initializer='random_uniform',activation='softmax'))
    #optimizer = tf.train.RMSPropOptimizer(0.01)
    
    model.compile(optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['acc'])
    #model.compile(loss='mse',
    #           optimizer=optimizer,
    #           metrics=['mae'])
    return model
def build_model_single():
    model = keras.Sequential();

    model.add(keras.layers.Dense(units=10, input_shape =( 784,),kernel_initializer='random_uniform',activation='softmax'));
    #optimizer = tf.train.RMSPropOptimizer(0.01)
    
    model.compile(optimizer='adam',
      loss='categorical_crossentropy',
      metrics=['acc'])
    #model.compile(loss='mse',
    #           optimizer=optimizer,
    #           metrics=['mae'])
    return model
        
if(True): #ENCODER 
    layer_configuration = [784,80]#121,100,81,64];
  #  layer_configuration = [784,100];

    for conf_num in range(1,len(layer_configuration)):
        
        w = None;
        encoder_validation_error = 0;
        if(conf_num > 0):
            #print(layer_configuration[:(conf_num +1)]);
            with open(data_utils.get_model_name_from_propeties(True, layer_configuration[:(conf_num +1)]), 'rb') as input:
                w = pickle.load(input)
               # for m in w.matrices:
                  #  print(m.shape);
                model = build_model(w,True);
                encoder_validation_error = w.train_summary[-1][1];
        else: 
            model = build_model_single();
        EPOCHS = 250
        
        #Store training stats
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
        
        
        history = model.fit(data_providers.train.data, to_categorical(data_providers.train.target), epochs=EPOCHS,
                           validation_split=0.1, verbose=0,shuffle=False,
                           callbacks=[early_stop]) #PrintDot()
        
        
        [loss, mae] = model.evaluate(data_providers.test.data, to_categorical(data_providers.test.target), verbose=0)

        print("Conf: " +  str(layer_configuration[:(conf_num +1)]) + " Encoder val. error: "  + str(encoder_validation_error) + " test acc "  + str(mae) +" Stopped at: " + str(len(history.history['val_loss'])) );
        
if(True): #MLP 
    layer_configuration = [784,121,100,81];
        
   
        
    model = build_model_normal(layer_configuration);
    EPOCHS = 1000
    
    #Store training stats
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30)
    
    
    history = model.fit(data_providers.train.data, to_categorical(data_providers.train.target), epochs=EPOCHS,
                       validation_split=0.1, verbose=0,shuffle=False,
                       callbacks=[early_stop]) #PrintDot()
    
    
    [loss, mae] = model.evaluate(data_providers.test.data, to_categorical(data_providers.test.target), verbose=0)

    print("Conf: " +  str(layer_configuration) + " test acc "  + str(mae) +" Stopped at: " + str(len(history.history['val_loss'])) );
