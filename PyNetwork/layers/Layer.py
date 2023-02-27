import abc


"""
    Todo
    ----
    * sequential.train needs a verbose parameter
"""


class Layer(abc.ABC):
    """ Abstract class to be used when creating layers for the `sequential` class
    """

    @abc.abstractmethod
    def build(self, previous_output_shape):
        """ When the build method is called in the `sequential` class it invokes this
            method. This allows the for the given layer to initialise the required variables.
        """
        pass

    @abc.abstractmethod
    def predict(self, z, output_only=True, pre_activation_of_input=None):
        """ When the predict method is called in the `sequential` class it invokes this
            method. This method is to perform the forward propagation of this current layer

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called

            Notes
            -----
            Most classes that will inherent `sequential_layer` will have an associated activation
            function.
            If `output_only = True` then this method is to return only the post-activated
            output.
            If `output_only = False` then this method is will return the pre-activated and post-activated
            output, in that order.
        """
        pass

    @abc.abstractmethod
    def get_delta_backprop_(self, g_prime, new_delta, prev_z):
        """ Returns the associated back propagation 'delta' for this layer

            Parameters
            ----------
            g_prime : np.array
                Should be the derivative of the output of the previous layer, g'_{k-1}(a^{k-1}_{m,j})
            new_delta : np.array
                The delta for this layer, delta^k_{m, j}
            prev_z : np.array
                The input for this layer, z^{n-1}

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called

            Notes
            -----
            Generally speaking, the layer does not need to be built in order for this method
            to work correctly. So perhaps this should be a static method of this class, but I'm not
            too sure about that yet. But until then, NotBuiltError will be raised unless it has
            been built
        """
        pass

    @abc.abstractmethod
    def get_weight_grad_(self, delta, prev_z):
        """ Returns the gradient for the bias and weight gradients, in that order.

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called
        """
        pass

    @abc.abstractmethod
    def update_parameters_(self, bias_updates, weight_updates):
        """ Once all the gradients have been calculated, this method will be called
            so the current layer can update it's weights and biases

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called
        """
        pass

    @abc.abstractmethod
    def get_weights(self):
        """ Returns the weights/filter and bias of this layer

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called
        """
        pass

    @abc.abstractmethod
    def summary_(self):
        """ Returns a tuple of strings that should identify the class.
            The 0th argument - The type of layer and the filter/weight shape
            The 1st argument - The output of the layer

            Raises
            ------
            NotBuiltError
                If the instance has not been built yet, i.e. `build` method must be called
        """

    @property
    @abc.abstractmethod
    def activation_function_(self):
        """ Returns the activation function of this layer
        """
        pass
