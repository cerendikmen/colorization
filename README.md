# colorization

cifar-10-python/cifar-10-batches-py contains CIFAR-10 data

foamliu (which comes from the Colorful-Image-Colorization.tar.gz file) contains code from foamliu's implementation of colorization. We were able to reimplement his code and run some experiments on them. We also modified his code to work on the CIFAR-10 imagenet dataset instead of the ImageNet data set it was originally designed for.

from_scratch is our attempt to re-implement Richard Zhang, Phillip Isola, Alexei A. Efros model by rewritting their code completely from scratch using python and tensorflow. This code is incomplete.

loss-function folder contains code we wrote in our attempt to run the original Caffe colorization implementation by Richard Zhang, Phillip Isola, Alexei A. Efros. Their code has been removed. The folder only contains the pieces of python code we wrote ourselves in an attempt to make their model work. 

nilboy folder contains nilboy's implementation of colorization with the last version of modifications we made. Our attempt to implement this version failed due to memory consumption errors.

Colorful-Image-Colorization.tar.gz file contains foamliu's code with our initial edits.
colorization-tf.tar.gz file contains nilboy's code with our initial edits.
working.tar.gz contains modifications to foamliu's code so that it works on CIFAR-10.
