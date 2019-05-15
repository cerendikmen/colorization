
# Data generation

import numpy as np;
import matplotlib.pyplot as plt
import random
from src.lab_1.neural_networks import SingleLayerNetwork,TwoLayerPerceptron,\
    ErrorEvaluatorMeanSquare, ErrorEvaluatorMissClassificationsArray
# Creates two classes of x,y coordinates and labels them.
from src.lab_1 import math_utils, plot_utils, neural_networks
from src.lab_1.data_utils import *;


random.seed(3333);
# Plot data
train_test_set = getTestBinaryTrainTestSet(True);
#train_test_set.plot_classes();

if(False): # boundary?

    s = getTestBinaryTrainTestSet(True);

    nepoch = 2000;

    eta =0.0001;
   
    nn = SingleLayerNetwork(2,1,True);
    error = nn.train_binary(nepoch,eta,s)[-1];
    
    plot_utils.plot_decision_boundary(train_test_set.test_set.observation, train_test_set.test_set.target, nn.W,'B');
    #plt.axis([-1.5,2.5,-1,1])
    print(error);
    #plt.legend(loc='upper right')
    #plt.show();
    plt.axis([-2,2,-2,2]);
    s.plot_classes();

    s.plot_classes();

nepoch = 1000;
if(False): #Delta vs percpetron 
    etas = [0.001,0.0001,0.00001]
    colors = ["green","red","blue"];
    for i in range(0,len(etas)):
        eta = etas[i];
        color = colors[i];
        nn = SingleLayerNetwork(2,1,True);
        errors = nn.train_binary(nepoch,eta,train_test_set,False,True);
        plt.plot(range(0, nepoch+1),errors,label='Delta rule  Eta = ' + str(eta),c=color)
        print(errors[-1]);
        nn = SingleLayerNetwork(2,1,True, nn.W_start);
        errors = nn.train_binary(nepoch,eta,train_test_set,False,False);
        print(errors[-1]);
        plt.plot(range(0, nepoch+1),errors,label='Perc. rule  Eta = ' + str(eta),c=color,linestyle='dashed')
        plt.xlabel("Number of epochs (batch learning iterations)")
        plt.ylabel("Number of misclassifications");
    plt.axis([0,100,0,1]);
    plt.legend(loc='upper right')
    plt.show();
if(False):
    
    etas = [0.001,0.0001,0.00001]
    colors = ["green","red","blue"];
    for i in range(0,len(etas)):
        eta = etas[i];
        color = colors[i];
        nn = SingleLayerNetwork(2,1,True);
        errors = nn.train_binary(nepoch,eta,train_test_set,False);
        plt.plot(range(0, nepoch+1),errors,label='Batch learn.  Eta = ' + str(eta),c=color)
      
        nn = SingleLayerNetwork(2,1,True, nn.W_start);
        errors = nn.train_binary(nepoch,eta,train_test_set,True);
        plt.plot(range(0, nepoch+1),errors,label='Seq. learn.  Eta = ' + str(eta),c=color,linestyle='dashed')
        plt.xlabel("Number of epochs")
        plt.ylabel("Number of misclassifications");
    plt.axis([0,40,0,1]);
    plt.legend(loc='upper right')
    plt.show();

if(False):
    eta = 0.001;
    train_test_set = getTestBinaryTrainTestSet(True,[0.5,0]); #[-0.1,0],[0.5,0]
    #train_test_set.plot_classes();
    nn = SingleLayerNetwork(2,1,True);
    errors = nn.train_binary(nepoch,eta,train_test_set,False);
    print("With bias, error is: " + str(errors[-1]))
    nn = SingleLayerNetwork(2,1,False);
    errors = nn.train_binary(nepoch,eta,train_test_set,False);
    print("Without bias, error is: " + str(errors[-1]) )

if(False): # Different tests on non linearly separated data
    train_test_set = getTestBinaryTrainTestSet(False);
    #Remove samples
    #25 % from each class
  
        
    set_of_sets = get_wanted_set_permutations(train_test_set);

    eta =0.001;
    for key in set_of_sets:   
        s = set_of_sets[key];  
        nn = SingleLayerNetwork(2,1,True);
        error = nn.train_binary(nepoch,eta,s)[-1];
        print('------')
        print(key + " Error: " + str(error)  +" SS: " +str(math_utils.getSensitivtyAndSpecificity(nn,s.test_set)));
        plot_utils.plot_decision_boundary(train_test_set.test_set.observation, train_test_set.test_set.target, nn.W,key);
    plt.axis([-1,1,-1,1])
    plt.legend(loc='upper right')
    train_test_set.plot_classes();

            
#Two layer
train_test_set = getTestBinaryTrainTestSet(False);
alpha = 0.4;
if(True): #Delta vs percpetron 
    etas = [0.01,0.001,0.0001]
    number_of_hidden_nodes_arr = [256,128,64,32]
    colors = ["green","red","blue"];
    line_styles = [":","-.","--","-"];
    nepoch = 300;
    for i in range(0,len(etas)):
        for j in range(0, len(number_of_hidden_nodes_arr)):
            eta = etas[i];
            color = colors[i];
            ls = line_styles[j];
            number_of_hidden_nodes = number_of_hidden_nodes_arr[j];
            nn = TwoLayerPerceptron(2,number_of_hidden_nodes,1,True);
            errors =  nn.train_binary(nepoch,eta,alpha,train_test_set,False);
            plt.plot(range(0, nepoch+1),errors,label='Eta= ' + str(eta) + 'hidden nodes= ' + str(number_of_hidden_nodes), c=color, linestyle = ls);
            plt.xlabel("Number of epochs (batch learning iterations)")
            plt.ylabel("Number of misclassifications");
            print("Eta= " +  str(eta) + ' nhn: ' + str(number_of_hidden_nodes) + " errors: " + str(errors[-1]))
    plt.axis([0,300,0,250]);
    plt.legend(loc='upper right')
    plt.show();
    
if(False): #Different samples 2 layer
    eta = 0.001;
    number_of_hidden_nodes = 32;

    
    set_of_sets = get_wanted_set_permutations(train_test_set, True);
    colors = ["black","green","red","blue","purple"];
    count = 0;
    set_of_sets.pop("Default");
    for key in set_of_sets:
        nn = TwoLayerPerceptron(2,number_of_hidden_nodes,1,True);
        (errors,errors_train) =  nn.train_binary(nepoch,eta,alpha,set_of_sets[key],False, True);
        plt.plot(range(0, nepoch+1),errors,label='Val. error, Sample type= ' + key, c = colors[count], linestyle="--");
        plt.plot(range(0, nepoch+1),errors_train,label='Train error, Sample type= ' + key, c = colors[count],);
   
        count += 1;
    plt.axis([0,500,0,1]);
    plt.xlabel("Number of epochs (batch learning iterations)")
    plt.ylabel("Number of misclassifications");
    plt.legend(loc='upper right')
    plt.show();
    


if(False): #Different samples 2 layer
    nepoch = 300;
    eta = 0.001;
    alpha = 0.9;
    number_of_hidden_nodes = 32;
    
  
    set_of_sets = get_wanted_set_permutations(train_test_set, True);
    colors = ["black","green","red","blue","purple"];
    count = 0;
    set_of_sets.pop("Default");
    for key in set_of_sets:
        nn = TwoLayerPerceptron(2,number_of_hidden_nodes,1,True);
        (errors) =  nn.train_binary(nepoch,eta,alpha,set_of_sets[key],False);
        nn = TwoLayerPerceptron(2,number_of_hidden_nodes,1,True,nn.A_start,nn.B_start);

        (errors_seq) =  nn.train_binary(nepoch,eta,alpha,set_of_sets[key],True);

        plt.plot(range(0, nepoch+1),errors,label='Batch learn.. error, Sample type= ' + key, c = colors[count]);
        plt.plot(range(0, nepoch+1),errors_seq,label='Seq. learn., Sample type= ' + key, c = colors[count], linestyle="--");
   
        count += 1;
    plt.axis([0,300,0,1]);
    plt.xlabel("Number of epochs (batch learning iterations)")
    plt.ylabel("Number of misclassifications");
    plt.legend(loc='upper right')
    plt.show();
    

if(False): #Attempt plot boundary
    nepoch = 1000;
    eta = 0.01;
    train_alpha = 0.9;
    number_of_hidden_nodes = 80;
    train_test_set = getTestBinaryTrainTestSet(False);
    x_min = min(train_test_set.train_set.observation[:,0]);
    x_max = max(train_test_set.train_set.observation[:,0]);
    nx = 100;
    ny = 100;
    extra = 1;
    x_range = np.linspace(x_min-extra,x_max+extra,nx,True);
    y_min = min(train_test_set.train_set.observation[:,1]);
    y_max = max(train_test_set.train_set.observation[:,1]);
    y_range = np.linspace(y_min-extra,y_max+extra,ny,True);   

    nn = TwoLayerPerceptron(2,number_of_hidden_nodes,1,True);
    (errors) =  nn.train_binary(nepoch,eta,train_alpha,train_test_set,False);
    for x in x_range:
        for y in y_range:

            pred = np.sign(nn.predict(np.reshape(np.array([x,y,1]),[3,1])));
            if(pred == -1):
                plt.plot(x,y,c="blue",marker='s', alpha=0.1);
            if(pred == 1):
                plt.plot(x,y,c="red",marker='s', alpha=0.1);
    plt.axis([x_min,x_max,y_min,y_max]);
    plt.show();
    train_test_set.plot_classes();

    
if(True): #Encoder
    set_of_sets = [];
    nepoch = 15000;
    eta = 0.1;
    train_alpha = 0.9;
    number_of_hidden_nodes = 3;
    patterns = [];
    targets =[];
    n = 8;
    new_arr = [];
    for i in range(0,n):
        new_arr = np.multiply(np.ones([1,n]),-1)
        new_arr[0,i] = 1;

        patterns.append(new_arr);
        targets.append(new_arr);
   # patterns.append(new_arr);
    #targets.append(new_arr);

    patterns = np.vstack((patterns));  
    targets = np.vstack((targets));  

    train = ObservationTarget(patterns, targets);
    test = train;
   
    nn = TwoLayerPerceptron(8,number_of_hidden_nodes,8,False);
    (errors, error_train) = nn.train(ErrorEvaluatorMissClassificationsArray, nepoch,eta,alpha, train, test,False, True,True);
  
    plt.plot(range(0, nepoch+1),errors,label='Val. error', linestyle='-');
    
    #plt.plot(range(0, nepoch+1),error_train,label='Train. error');
    print('End error: ' + str(errors[-1]))
    plt.xlabel("Epoch");
    plt.ylabel("Number of errors");
    plt.legend();
    plt.show();

    print(np.round(math_utils.phi(np.dot(nn.A, patterns.T)),2));
    
if(False): # FUNC APPRX
   
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.subplots_adjust(wspace=0.2, right=0.86)
    if(True): #plot?
        plt.subplot(1, 2, 1)
        dx = 0.5;
        dy = 0.5;
        y, x = np.mgrid[slice(-5, 5 + dy, dy),
                    slice(-5, 5 + dy, dy)]
       
       
        x = np.reshape(x,[np.size(x,0)*np.size(x,1),1]);
        y = np.reshape(y,[np.size(y,0)*np.size(y,1),1]);
        z = math_utils.fun_func(x,y);
        
        plot_utils.surface_plot_from_xyz(x, y, z,-0.5,0.5, False);
        
    x = np.linspace(-5, 5, 21);
    y = np.linspace(-5, 5, 21);
    nepoch = 1000;
    etas = [0.001,0.005];
    train_alpha = 0.9;
    number_of_hidden_nodes = 62;
    data = [];
    for a in x:
        for b in y:
  
            data.append(np.reshape(np.array([a,b,math_utils.fun_func(a, b)]),[1,3]));
    data = np.vstack(data);
    all_pattern = data[:, :-1];
    all_target = data[:, -1:];
    all_indices = range(0, np.size(data,0));
    percentages = [0.8];#0.2,0.4,0.6,0.8#,0.4,0.6,0.8];
    nn = None;
    for eta in etas:
        for i in range(25,26):
            if(np.mod(i, 5) != 0):
                continue;
            for p in percentages:
                sample_indices = random.sample(all_indices, int(round(len(all_indices)*p)));
                sample_data = [];
                for sample_index in sample_indices:
                
                    sample_data.append(data[sample_index,:]);
    
                sample_data = np.vstack(sample_data);
                pattern = sample_data[:, :-1];
                target = sample_data[:, -1:];
                neural_networks.q(pattern);
                
                train = ObservationTarget(pattern, target);
                test =  ObservationTarget(all_pattern, all_target);
                if(nn == None):
                    nn = TwoLayerPerceptron(np.size(pattern,1),i,1,True);
                else:
                    nn = TwoLayerPerceptron(np.size(pattern,1),i,1,True, nn.A,nn.B);
    
                (errors, error_train) = nn.train(ErrorEvaluatorMeanSquare,nepoch,eta,alpha, train, test,False, True,True);
              
               # plt.plot(range(0, nepoch+1),errors,label=' Val. error HN=' + str(i) + ' p=' + str(p) + ' eta='+str(eta) , linestyle='--');
               # plt.plot(range(0, nepoch+1),error_train,label=' Train. error HN=' + str(i) + ' p=' + str(p)+ ' eta='+str(eta));
                print('HN: ' + str(i) + ' End error: ' + str(min(errors)))
   # plt.xlabel("Number of epochs (batch learning iterations)")
   # plt.ylabel("Std error");
   # plt.legend(loc='upper right')
    #plt.axis([0, 300, 0.25, 0.3])
    #plt.show();
    if(True): #plot?
        plt.subplot(1, 2, 2)
        dx = 0.5;
        dy = 0.5;
        y, x = np.mgrid[slice(-5, 5 + dy, dy),
                    slice(-5, 5 + dy, dy)]
      
        x = np.reshape(x,[np.size(x,0)*np.size(x,1),1]);
        y = np.reshape(y,[np.size(y,0)*np.size(y,1),1]);
       
        z = nn.predict(np.hstack((x,y)), False, True).T;
        
       
        m = plot_utils.surface_plot_from_xyz(x, y, z,-0.5,0.5,False);
    p = plt.figure(0);
   
    cbar_ax = fig.add_axes([0.89, 0.1, 0.05, 0.8])
    
    p.colorbar(m,cax=cbar_ax);
    plt.show();
# plot solution
#x = np.array(range(int(np.min(train_test_set.test_set.observation[:,0])-1),int(np.max(train_test_set.test_set.observation[:,0])+1)))  


#plot decision boundary
    
   

