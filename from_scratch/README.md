# instructions to run from_scratch implementation

1) Create directory "cifar-10-npy"
2) Run "Create_Labels_and_Data_NPY_Files.py"
3) Run "CNN_Model.py"
4) Predictions can be found at "cifar-10-npy/test_data_predictions.npy"
5) To visualize these predictions, use jupyter notebook with "Rebuild_Images_Predictions.ipynb".
5) To visualize base truth, use jupyter notebook with "Rebuild_Images.ipynb". Note that base truth here refers to images after they have been convert to LAB color space and then into bins, and not the original CIFAR file. However, our tests have shown that there is minimal loss in color due to these conversions alone.