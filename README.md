# PyNetwork
Lightweight implementation for deep learning using Numpy. 

It implements four standard layers: (1) BatchNorm, (2) Dense, (3) Convolutional, and (4) Flatten layers. It also comes with three optimizers: (1) SGD (with and without Nesterov Acceleration), (2) RMSProp, and (4) Adam. 

### Your Job
PyNetwork implements all the necessary building blocks required for neural networks. Your job is to 

1) Get it working on the GPU. To do this, use PyOpenCL to rewrite the layers, optimizers, and functions so they can work on the GPU.
2) Implement l1 and l2 regularization for the layers. An example of l1 & l2 regularization can be seen in the dense layer.
3) Ability to set weights as being trainable or non-trainable. This is required to explore the lottery ticket hypothesis. (This is much easier than it sounds)

### Extensions
There are many more things you can implement. From easiest to hardest,
1) Different initializations for the layers (see [link](https://www.tensorflow.org/api_docs/python/tf/keras/initializers))
2) Different optimizers (see [link](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers))
3) Different layers (see [link](https://www.tensorflow.org/api_docs/python/tf/keras/layers))

### Helpful Tips
1) To make sure the output from your GPU code is correct, you should check them against the output from the given CPU code. An example can be seen in the "Check Dense Layer" notebook.
2) In the sequential class, we use `x_train[index[start:end]]` to perform training on the batches. But this is not a contiguous array, and PyOpenCL will raise an error.