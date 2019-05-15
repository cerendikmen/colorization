class WeightMatrices: 
    matrices = [];
    biases = [];
    train_summary = [];
    def __init__(self, matrices, biases, train_summary):
        self.matrices =matrices;
        self.biases  = biases;
        self.train_summary = train_summary;