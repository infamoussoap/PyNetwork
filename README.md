# PySoap
Light weight implementation for deep learning. 

# Why PySoap
While Tensorflow is the start of the art software for machine learning, it requires a heavy amount of hardware space to be installed. 
PySoap, primarily developed for my own research, is a light weight implementation of fully connected and convolutional neural networks. 
The only requirements are numpy, abc, inspect, and h5py.

# What is in PySoap
PySoap, currently, implements Dense, Conv_2D, and BatchNorm layers. It also features many different optimizers such as Adam, SGD, Adagrad, etc.

# PySoap Interface
PySoap is created to feel much like creating a Sequential model using Keras. As such, users of Keras will feel at home when using PySoap
