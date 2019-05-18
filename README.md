# colorization

cifar-10-python/cifar-10-batches-py contains CIFAR-10 data

foamliu (which comes from the Colorful-Image-Colorization.tar.gz file) contains code from foamliu's implementation of colorization. We were able to reimplement his code and run some experiments on them.

from_scratch is our attempt to re-implementation by Richard Zhang, Phillip Isola, Alexei A. Efros by rewritting the code completely from scratch using python and tensorflow

loss-function folder contains code we wrote in out first attempt to run the original Caffe colorization implementation by Richard Zhang, Phillip Isola, Alexei A. Efros. Their code has been removed from, this folder only contains the pieces of python implementation we tried.

nilboy folder (which comes from the colorization-tf.tar.gz file) contains code from nilboy's implementation of colorization. This contains mostly their code. Our attempt to implement this version failed.
