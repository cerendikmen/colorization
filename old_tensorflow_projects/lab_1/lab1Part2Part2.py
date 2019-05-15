import matplotlib.pyplot as plt;
import tensorflow as tf;
import numpy as np;
from tensorflow import keras
from tensorflow.keras import regularizers
import time

from tensorflow.python.keras.layers.core import Dense
import random;
random.seed(333);
np.random.seed(333);
tf.set_random_seed(333);

# init data


def q(x):
    if(np.ndim(x) == 1):
        return len(x);
    print((np.size(x,0),np.size(x,1)));
    

def get_data(sigma):
    t = range(0,1506);

    x = [None]*len(t);
    x[0] = 1.5;
    def x_at_t_plus_1(x, t):
        return get_x_at_t(x,t) + ((0.2*get_x_at_t(x,t-25))/(1 + get_x_at_t(x,t-25)**10) - 0.1*get_x_at_t(x,t));
    def get_x_at_t(x,t):
        if t < 0:
            return 0;
        return x[t];
    ti = t[:len(t)-1];
    for i in ti:
        x[i+1] = x_at_t_plus_1(x,i);
    for i in range(0, len(x)):
        x[i] = np.add(x[i],  np.random.normal(0,sigma,1)[0]);
    t_part = range(301,1501);
    inputs = [];
    outputs = [];
    for i in t_part:
        inputs.append([x[i-20], x[i-15],x[i-10],x[i-5], x[i]]);
        outputs.append(x[i+5]);
    outputs = np.vstack(outputs);
    validation_end_index = 1119;
    inputs  = np.vstack(inputs);

    inputs_train = inputs[:validation_end_index, :];
    inputs_test = inputs[validation_end_index:, :];
    outputs_train = outputs[:validation_end_index];
    outputs_test = outputs[validation_end_index:];
    # Tensor flow
    mean = inputs_train.mean(axis=0)
    std = inputs_train.std(axis=0)
    inputs_train = (inputs_train - mean) / std
    inputs_test = (inputs_test - mean) / std
    return ((std, mean,inputs_train,outputs_train, inputs_test,outputs_test));


# Parameters


learning_rate = 0.001
number_of_hidden_nodes = [10,30,75];
regularization = [0,0.000001,0.00001];#[0.0001];#[0.01,0];
ls_reg = ["-","--"];
color_nodes = ["r","b","g"];

count = -1;
n_count = -1;
sigmas = [0.03, 0.09, 0.18];
if(False):
    for sigma in sigmas:
        (std, mean,inputs_train,outputs_train, inputs_test,outputs_test) = get_data(sigma);
     
        for n in number_of_hidden_nodes:
            n_count += 1;
            reg_count = -1;
            for r in regularization:
                reg_count += 1;
                count += 1;
                def build_model(r,n):
                    model = keras.Sequential([
                    keras.layers.Dense(10, activation=tf.nn.sigmoid,
                                       input_shape=(inputs_train.shape[1],),
                                       kernel_regularizer=regularizers.l2(r)
                                       ),
                     keras.layers.Dense(n, activation=tf.nn.sigmoid,
                                       input_shape=(10,),
                                       kernel_regularizer=regularizers.l2(r)
                                       ),
                                       
                    keras.layers.Dense(1, 
                                       input_shape=(inputs_train.shape[1],),
                                       kernel_regularizer=regularizers.l2(r))
                    ])
                  
                    optimizer = tf.train.RMSPropOptimizer(learning_rate)
                    
                    model.compile(loss='mse',
                                optimizer=optimizer,
                                metrics=['mae'])
                    return model
                
                model = build_model(r,n)
        
                
                EPOCHS = 1000
                
                # Store training stats
                early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
                
            
                history = model.fit(inputs_train, outputs_train, epochs=EPOCHS,
                                    validation_split=0.2, verbose=0,shuffle=False,
                                    callbacks=[early_stop]) #PrintDot()
            
                
                [loss, mae] = model.evaluate(inputs_test, outputs_test, verbose=0)
                print("Val mean square error: "  + str(round(history.history['val_loss'][-1],4)) +" Stopped at: " + str(len(history.history['val_loss']))  +  " reg: "  +  str(r) + " #hn: " + str(n) + " sigma:" + str(sigma));

if(True):
    iteration = [ (30, 0), (30,0.000001), (10,0.000001), (10, 0.0001)]# last setting is two layer , #
   
    for sigma in sigmas:
        (std, mean,inputs_train,outputs_train, inputs_test,outputs_test) = get_data(sigma);
        for i in range(0,len(iteration)):
            (n,r) = iteration[i];
            threeLayer = True;
            if i == len(iteration)-1:
                threeLayer = False;
            def build_model(r,n,threeLayer):
                model = None;
                
                if(threeLayer):

                    model = keras.Sequential([
                    keras.layers.Dense(10, activation=tf.nn.sigmoid,
                                       input_shape=(inputs_train.shape[1],),
                                       kernel_regularizer=regularizers.l1(r)
                                       ),
                     keras.layers.Dense(n, activation=tf.nn.sigmoid,
                                       input_shape=(10,),
                                       kernel_regularizer=regularizers.l1(r)
                                       ),
                                       
                    keras.layers.Dense(1, 
                                       input_shape=(inputs_train.shape[1],),
                                       kernel_regularizer=regularizers.l1(r))
                    ])
                else:
                    model = keras.Sequential([
                     keras.layers.Dense(n, activation=tf.nn.sigmoid,
                                   input_shape=(inputs_train.shape[1],),
                                   kernel_regularizer=regularizers.l1(r)
                                   ),
                                   
                    keras.layers.Dense(1, 
                                       input_shape=(inputs_train.shape[1],),
                                       kernel_regularizer=regularizers.l1(r))
                    ])
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
                
                model.compile(loss='mse',
                            optimizer=optimizer,
                            metrics=['mae'])
                return model
            
            model = build_model(r,n,threeLayer)
    
            
            EPOCHS = 1000
            
            # Store training stats
            #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
            
            start_time = time.time()
            history = model.fit(inputs_train, outputs_train, epochs=EPOCHS,
                                validation_split=0.2, verbose=0,shuffle=False,
                                #callbacks=[early_stop]
                                ) #PrintDot()
        
            end_time = time.time()
            total_time = end_time - start_time;
            [loss, mae] = model.evaluate(inputs_test, outputs_test, verbose=0)
            
            if(threeLayer):
                print(str(round(loss,4)) + " & "  +  str(r) + " & " + "10-"+str(n) + " & " + str(sigma) + "\\\ \hline");
            else:
                print(str(round(loss,4)) + " & "  +  str(r) + " & " + str(n) + " & " + str(sigma) + "\\\ \hline");


            #print("Val mean square error: "  + str(round(loss,4)) +" Stopped at: " + str(len(history.history['val_loss']))  +  " reg: "  +  str(r) + " #hn: " + str(n) + " Three layers? " + str(threeLayer) + " sigma:" + str(sigma) + " time: " + str(total_time) );
    