# PyNetwork
Lightweight implementation for deep learning using Numpy. 

It implements four standard layers: (1) BatchNorm, (2) Dense, (3) Convolutional, and (4) Flatten layers. It also comes with three optimizers: (1) SGD (with and without Nesterov Acceleration), (2) RMSProp, and (4) Adam. 

### Your Job
PyNetwork implements all the necessary building blocks required for neural networks. Your job is to 
1) Get it working on the GPU. To do this, use PyOpenCL to rewrite the layers, optimizers, and functions so they can work on the GPU.
2) Implement l1 and l2 regularization for the layers. An example of l1 & l2 regularization can be seen in the dense layer.
3) Ability to set weights as being trainable or non-trainable. An example can be seen in the dense layer. This is required to explore the lottery ticket hypothesis. (This is much easier than it sounds)

### Extensions
There are many more things you can implement. From easiest to hardest,
1) Different initializations for the layers (see [link](https://www.tensorflow.org/api_docs/python/tf/keras/initializers))
2) Different optimizers (see [link](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers))
3) Different layers (see [link](https://www.tensorflow.org/api_docs/python/tf/keras/layers))
4) [Non-sequential network](https://machinelearningmastery.com/keras-functional-api-deep-learning/), these are multiple networks which run in parallel and can be split over multiple GPUs. These parallel networks are then merged together through a concatenate layer. Essentially, this transforms the linear path of the sequential model into a graph like structure. 

### Helpful Tips
1) Start off with implementing the Dense layer into the GPU. Once you have done that, implementing the code for the other parts of the code should be easier.
2) The convolutional layer is the most difficult to code. You do not need to implement this, but I will be very impressed if you can.
3) To make sure the output from your GPU code is correct, you should check them against the output from the given CPU code. An example can be seen in the "Check Dense Layer" notebook.
4) In the sequential class, we use `x_train[index[start:end]]` to perform training on the batches. But this is not a contiguous array, and PyOpenCL will raise an error.

Colab Links 
- Compilers: https://colab.research.google.com/drive/1sCuZDfE_hTqZFDyYKkX1DvZLlZubE48A?usp=sharing
- Cupy: https://colab.research.google.com/drive/13ulTBbtxe-Xs35VxSjd1mu0j_zlQpVCy?usp=sharing
- Why GPUs: https://colab.research.google.com/drive/1NC4rpujkRiBSesPAJ_DxRcQns7wU0Pb-?usp=sharing
- PyOpenCL: https://colab.research.google.com/drive/15yk8JbY-GadZhyUDyb1MLAokatYhJ0PQ?usp=sharing
