import time
import datetime

import tensorflow as tf


from src.lab_4.rbm_base import RBMBase

class RBMTrainByPairs(RBMBase):
    def build_model(self):
        self._create_placeholders()
        self._create_variables()

        # pass input through previous layers to construct right input
        layers_qtty = self.params.get('layers_qtty', 1)
        inputs = self.inputs
        for layer_no in range(layers_qtty - 1):
            hid_layer_no = layer_no + 1
            hid_probs, hid_states = self._sample_hidden_from_visible(
                inputs, hid_layer_no)
            if self.bin_type:
                inputs = hid_states
            else:
                inputs = hid_probs

        # now enable RBM for last two layers
        self.updates = []
        layer_from = layers_qtty - 1
        layer_to = layers_qtty
        tmp_res = self.rbm_block(
            inputs=inputs, layer_from=layer_from, layer_to=layer_to)
        updates, vprob_last, hprob_last, hstate_last = tmp_res
        self.updates.extend(updates)

        self.encoded_array = hstate_last

        if self.bin_type:
            last_out = hstate_last
        else:
            last_out = hprob_last
        for vis_layer_no in list(reversed(range(layers_qtty))):
            last_out = self._sample_visible_from_hidden(
                hidden_units=last_out, vis_layer_no=vis_layer_no)
        self.reconstruction = last_out

        # add some summaries
        self.cost = tf.reduce_mean(tf.square(tf.subtract(self.inputs, self.reconstruction)));
        self.square_error = self.cost;#tf.losses.mean_squared_error(labels=self.inputs, predictions=tf.round(self.reconstr_prob));#, weights, scope, loss_collection, reduction)
        self.free_energy = tf.Variable(0.);
        for vis_layer_no in list(reversed(range(layers_qtty))):
            hid_layer_no = vis_layer_no+1;
            
            weights = getattr(self, 'W_%d_%d' % (vis_layer_no, hid_layer_no))
            bias_a = getattr(self, 'bias_%d' % vis_layer_no)
            bias_b = getattr(self, 'bias_%d' % hid_layer_no)
            bias_a =  tf.reshape(bias_a,[tf.size(bias_a),1]);
            bias_b =  tf.reshape(bias_b,[tf.size(bias_b),1]);
            #p_val = tf.nn.sigmoid(x_val);
            #log_p_val = tf.log(p_val);
            #one_min_p_val = tf.subtract(1., p_val);
            #two_first_terms = tf.subtract(tf.multiply(-1.,tf.reduce_sum(tf.multiply(self.inputs,bias_a),axis=1)),tf.reduce_sum(tf.multiply(p_val,x_val),axis=1));
            #second_term = tf.reduce_sum(tf.add(tf.multiply(p_val, log_p_val), tf.multiply(one_min_p_val, tf.log(one_min_p_val))));
            #self.free_energy = tf.add(self.free_energy, tf.reduce_mean(tf.add(two_first_terms, second_term)));
            #self.free_energy = tf.add(self.free_energy, tf.reduce_mean(tf.subtract(tf.multiply(-1., tf.multiply(self.inputs,bias_a)), tf.reduce_sum(tf.log(tf.add(1., tf.exp(tf.add(bias_b,tf.matmul(self.inputs, weights)))))))));
            layer_sum = tf.Variable(0.);
           
            #row = tf.reshape(self.inputs[0,:],[1, tf.size(self.inputs[0,:])]);
            #x_val = tf.add(bias_b,tf.transpose(tf.matmul(row,weights)));#tf.matmul( weights,row,transpose_a=True,transpose_b=True));
            #eee = tf.matmul(row,bias_a);
            #layer_sum = tf.add(layer_sum, tf.subtract(tf.multiply(-1., eee), tf.reduce_sum(tf.log(tf.add(1., tf.exp(x_val))))));
            #self.free_energy = tf.add(self.free_energy, tf.divide(layer_sum,tf.Variable(1.)));
            
            
            x_val = tf.transpose(tf.add(bias_b,tf.transpose(tf.matmul(inputs,weights))));#tf.matmul( weights,row,transpose_a=True,transpose_b=True));
           # print(x_val);
            eee = tf.matmul(inputs,bias_a);
          #  print(eee);
            layer_sum = tf.reduce_mean(tf.add(layer_sum, tf.subtract(tf.multiply(-1., eee), tf.reduce_sum(tf.log(tf.add(1., tf.exp(x_val)))))));
            self.free_energy = tf.add(self.free_energy, tf.divide(layer_sum,tf.Variable(1.)));
        tf.summary.scalar("train_loss", self.cost)
        self.summary = tf.summary.merge_all()


    def _get_restored_variables_names(self):
        """Get variables that should be restored from previous run for
        all layers but last one.
        """
        restore_dict = {}
        for layer_no in range(self.params['layers_qtty']):
            bias_name = "bias_%d" % layer_no
            restore_dict[bias_name] = getattr(self, bias_name)
            if layer_no > 0:
                w_name = "W_%d_%d" % (layer_no - 1, layer_no)
                restore_dict[w_name] = getattr(self, w_name)
        return restore_dict

    def _get_new_variables_names(self):
        """Get variables for last layer - it should be initialized in
        any case.
        """
        last_layer = self.params['layers_qtty']
        w_name = "W_%d_%d" % (last_layer - 1, last_layer)
        w = getattr(self, w_name)
        bias_name = "bias_%d" % last_layer
        bias = getattr(self, bias_name)
        new_vars = [w, bias]
        return new_vars

    trained_matrices = [];
    trained_biases = [];
    
    def train(self, output_summary = []):
        initial_params = dict(self.params)
        params = self.params
        epochs_per_pair = params['epochs'] // params['layers_qtty']
        for layers_qtty in range(1, initial_params['layers_qtty'] + 1):
            tf.reset_default_graph()
            print("Train layers pair %d and %d for %d epochs" % (
                layers_qtty - 1, layers_qtty, epochs_per_pair))
            self.params['epochs'] = epochs_per_pair
            self.params['layers_qtty'] = layers_qtty
            self.params['layers_sizes'] = initial_params['layers_sizes'][:layers_qtty + 1]
            self._train_layer_pair(output_summary,layers_qtty-1)


    def _train_layer_pair(self,output_summary,layer_num):
        self.build_model()
        prev_run_no = self.params.get('run_no', None)
        self.define_runner_folders()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            self.sess = sess

            if prev_run_no:
                print("Restore variables from previous run:")
                restore_vars_dict = self._get_restored_variables_names()
                for var_name in restore_vars_dict.keys():
                    print("\t%s" % var_name)
                restorer = tf.train.Saver(restore_vars_dict)
                restorer.restore(sess, self.saves_path)
                print("Initialize not restored variables:")
                new_variables = self._get_new_variables_names()
                for var in new_variables:
                    print("\t%s" % var.name)
                sess.run(tf.initialize_variables(new_variables))

            else:
                print("Initialize new variables")
                tf.global_variables_initializer().run()
           
            larger_val_error_counter = 0;
            last_validation_error = 10000000000;
            for epoch in range(self.params['epochs']):
                start = time.time()
                self._epoch_train_step(output_summary);
                

                time_cons = time.time() - start
                time_cons = str(datetime.timedelta(seconds=time_cons))

                if(output_summary[-1][1] >= last_validation_error):
                    larger_val_error_counter += 1;
                    if(larger_val_error_counter > self.params['early_stopping_number']):
                        print("Early stopping");
                        break;
                else:
                    larger_val_error_counter = 0;
                last_validation_error = output_summary[-1][1]; 
                print("Epoch: {a}, Val error {b}, Free energy train {c},Free energy validation {d}, Diff {e}".format(a=epoch, b=output_summary[-1][1],c=output_summary[-1][2],d=output_summary[-1][3],e = output_summary[-1][3]-output_summary[-1][2]))
            


            # Save all trained variables
            self.trained_matrices.append(getattr(self,"W_{a}_{b}".format(a = layer_num, b= layer_num +1)).eval());
            if(layer_num == 0):
                self.trained_biases.append(getattr(self,"bias_0").eval());
            self.trained_biases.append(getattr(self,"bias_{b}".format(b= layer_num +1)).eval());


            saver = tf.train.Saver()
            print(self.saves_path)
            saver.save(sess, self.saves_path)
            