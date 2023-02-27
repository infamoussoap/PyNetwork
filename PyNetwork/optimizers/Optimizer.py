import abc


class Optimizer(abc.ABC):
    """ Abstract class to be used when creating layers for the `sequential` class
    """

    @abc.abstractmethod
    def step(self, grad_dict):
        """ Returns the gradient update as defined by the algorithm

            Parameters
            ----------
            grad_dict : dict of int - np.array
                Dictionary of gradients, where keys represent the layer number and the
                corresponding value is the layer gradients

            Returns
            -------
            dict of int - np.array
                Dictionary of the gradients as scheduled by the algorithm. The keys represent the
                layer number, and the corresponding value will be the scheduled gradient

            Notes
            -----
            This function returns the value to subtract from the current parameters.
            Consider grad_dict as dS/da, with a the parameters of the network. Then to
            update the parameters of the network

            a = a - Optimizer.gradients(dS/da)

            Obviously the output is a dictionary, so you'll have to account for that.
        """
        pass

    @abc.abstractmethod
    def new_instance(self):
        """ Returns a new instance of the given Optimizer

            Returns
            -------
            Optimizer
        """

        pass
