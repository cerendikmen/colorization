import os
import numpy as np;
from src.lab_4.data_providers import TrainTest;

def get_model_name_from_propeties(is_encoder,hidden_nodes):
    return str(is_encoder) +"-"+str(hidden_nodes).replace("[", "_").replace("]","_")+".super_models"
def matrix_from_file_with_only_row(file_name,num_cols, file_ending = 'dat'):
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    path = os.path.join(fileDir, 'resources/' + file_name + '.'+file_ending);

    with open(path) as f:
        content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    content = content[0]; #one row;
    sp = content.split(",");
    fill = [];
    rows = [];
    for i in range(0, len(sp)):
        val  = float(sp[i]);
        if(i > 0 and np.mod(i,num_cols) == 0):
            rows.append(fill);
            fill = [];
        fill.append(val);
    rows.append(fill);

    rows = np.vstack(rows);
    return rows;

def matrix_from_file(file_name, file_ending = 'dat'):
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    path = os.path.join(fileDir, 'resources/' + file_name + '.'+file_ending);

    with open(path) as f:
        content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip().replace(";","") for x in content]
    rows = [];
    for c in content:
        spl = c.split(",");
        val_arr = [];
        for s in spl:
            val_arr.append(float(s));
        rows.append(val_arr);
    
    return np.vstack(rows);

def load_text_file_rows(file_name):
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    path = os.path.join(fileDir, 'resources/' + file_name + '.txt');

    with open(path) as f:
        content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip().replace("'","") for x in content]  
    return content;
def distort_binary(X, number): #X is list
    x = X.copy();
    index_arr = range(0, len(X));

    index_sub_arr = np.random.choice(index_arr, number, replace=False);
    for k in index_sub_arr:
        x[k] *= -1;
    return x;




def sort_data_set(train_data_bin,train_data_target, stop_if_uneven=True):
   
    buckets = [];
    indices_in_buckets = [];
    for i in range(0,10):
        buckets.append([]);
        indices_in_buckets.append(0);
    for i in range(0, np.size(train_data_bin,0)):
        buckets[int(train_data_target[i,:])].append(TrainTest(train_data_bin[i,:],train_data_target[i,:]));
    
    ret_bin = [];
    ret_target = [];
    bucket_index = 0;
    added_element = False;
    while(True):
        if(indices_in_buckets[bucket_index] < len(buckets[bucket_index])):
            ret_bin.append(buckets[bucket_index][indices_in_buckets[bucket_index]].data);
            ret_target.append(buckets[bucket_index][indices_in_buckets[bucket_index]].target);
            indices_in_buckets[bucket_index] = indices_in_buckets[bucket_index] + 1;
            added_element = True;
        elif(stop_if_uneven):
            break;
        bucket_index += 1;
        if(bucket_index == 10):
            if(not added_element):
                break;
            added_element = False;
            bucket_index = 0;
    return (np.vstack(ret_bin),np.vstack(ret_target));
    

    
            
    