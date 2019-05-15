import numpy as np;
class SimpleNNFunction:
    def __init__(self, weight_matrices, activation_function):
        self.weight_matrices = weight_matrices;
        
    def evaluate(self, x):
        x_val = np.add(x, self.weight_matrices.biases[0]);
        
      #  for i in len(self.weight_matrices.matrices):
      #      x_val = 
        