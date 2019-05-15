import matplotlib.pyplot as plt;
import tensorflow as tf;
import numpy as np;
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.python.keras.layers.core import Dense
import random;
random.seed(333);
np.random.seed(333);
tf.set_random_seed(333);

# init data
t = range(0,1506);
x = [None]*len(t);
x[0] = 1.5;

def q(x):
    print((np.size(x,0),np.size(x,1)));
    
def x_at_t_plus_1(x, t):
    return get_x_at_t(x,t) + ((0.2*get_x_at_t(x,t-25))/(1 + get_x_at_t(x,t-25)**10) - 0.1*get_x_at_t(x,t));
def get_x_at_t(x,t):
    if t < 0:
        return 0;
    return x[t];
ti = t[:len(t)-1];
for i in ti:
    x[i+1] = x_at_t_plus_1(x,i);
 

t_part = range(301,1501);

inputs = [];
outputs = [];
for i in t_part:
    inputs.append([x[i-20], x[i-15],x[i-10],x[i-5], x[i]]);
    outputs.append(x[i+5]);
outputs = np.vstack(outputs);
train_end_index = 800;
validation_end_index = 1000;
inputs  = np.vstack(inputs);
q(inputs);
inputs_train = inputs[:validation_end_index, :];
inputs_test = inputs[validation_end_index:, :];
outputs_train = outputs[:validation_end_index];
outputs_test = outputs[validation_end_index:];
# Tensor flow
mean = inputs_train.mean(axis=0)
std = inputs_train.std(axis=0)
inputs_train = (inputs_train - mean) / std
inputs_test = (inputs_test - mean) / std
# Parameters
if(True):
    plot_hist = False;
    plot_test_prediction = True;
    plot_validation = False;
    learning_rate = 0.001
    
    #error = 0.001817
    number_of_hidden_nodes = [10,30,50];
    regularization = [0.0001];#[0.01,0];
    ls_reg = ["-","--"];
    color_nodes = ["r","b","g"];

    count = -1;
    if(plot_hist):
        fig, axes = plt.subplots(nrows=1, ncols=len(regularization))
    n_count = -1;

    for n in number_of_hidden_nodes:
        n_count += 1;
        reg_count = -1;
        for r in regularization:
            reg_count += 1;
            count += 1;
            def build_model(r,n):
                model = keras.Sequential([
                keras.layers.Dense(n, activation=tf.nn.sigmoid,
                                   input_shape=(inputs_train.shape[1],),
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
            #Test mean square error: 0.009214341640472412 reg: 0.001 #hn: 10
            #Test mean square error: 0.012825215309858322 reg: 0.001 #hn: 30
            #Test mean square error: 0.01862157553434372 reg: 0.001 #hn: 50

            model = build_model(r,n)
            #model.summary();
            # Display training progress by printing a single dot for each completed epoch
            class PrintDot(keras.callbacks.Callback):
              def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0: print('')
                print('.', end='')
            
            EPOCHS = 1000
            
            # Store training stats
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
            
            history = model.fit(inputs_train, outputs_train, epochs=EPOCHS,
                                validation_split=0.2, verbose=0,shuffle=False,
                                callbacks=[early_stop]) #PrintDot()
            
            def plot_history(history):
                plt.xlabel('Epoch')
                plt.ylabel('Mean square error')
                #plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),label='Train Loss')
                plt.plot(history.epoch, np.array(history.history['val_loss']), label = "#HN="+str(n) + " $\lambda$="+str(r), linestyle=ls_reg[reg_count], c=color_nodes[n_count])
                plt.legend()
                
                plt.ylim([0, 0.5])
                plt.xlim([0, 50])

               # plt.show();
            
               
            if(plot_validation):
                plot_history(history)
            
            [loss, mae] = model.evaluate(inputs_test, outputs_test, verbose=0)
            print("Test mean square error: "  + str(loss) + " reg: " +  str(r) + " #hn: " + str(n) );
            if(plot_hist):
                plt.subplot(1,len(regularization),1 + count);
                plt.hist(np.abs(model.get_weights()[0]), bins=4)
                plt.xlabel("Weights, Reg.= " + str(r));
                plt.ylabel('Count');
            if(plot_test_prediction):
                prediction = model.predict(inputs_test);
               
                plt.plot(t_part[len(t_part) - len(outputs_test):],outputs_test,label="Time series");
                plt.plot(t_part[len(t_part) - len(outputs_test):],prediction, label="Prediction");
                plt.xlabel("Time");
                plt.ylabel("Value");
                plt.legend();
            
    plt.show();





