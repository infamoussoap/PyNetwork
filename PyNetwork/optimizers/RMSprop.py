import numpy as np

from PyNetwork.optimizers import Optimizer


class RMSprop(Optimizer):
    """ Adam optimiser

        Attributes
        ----------
        learning_rate : float
            The learning rate or step size to take in the gradient given by adam
        rho : float
            Decay rate for the moving average. Must be 0 < b1 < 1
        e : float
            Arbitrarily small float to prevent division by zero error

        v : dict of int - np.array
            Stores the moving average of the second raw momentum
    """

    def __init__(self, learning_rate=0.001, rho=0.9, e=1e-7):
        """ Initialise attributes of Adam Optimiser

            Parameters
            ----------
            learning_rate : float, optional
            rho : float, optional
            e : float, optional
        """
        self.learning_rate = learning_rate
        self.rho = rho
        self.e = e

        self.v = None

    def step(self, grad_dict):
        """ Returns the gradients as scheduled by RMSprop

            Parameters
            ----------
            grad_dict : dict of int - np.array
                Dictionary of gradients, where keys represent the layer number and the
                corresponding value is the layer gradients

            Returns
            -------
            dict of int - np.array
                Dictionary of the gradients as scheduled by RMSprop. The keys represent the
                layer number, and the corresponding value will be the scheduled gradient

            Notes
            -----
            This function returns the value to subtract from the current parameters.
            Consider grad_dict as dS/da, with a the parameters of the network. Then to
            update the parameters of the network

            a = a - RMSprop.step(dS/da)

            Obviously the output is a dictionary, so you'll have to account for that.
        """
        if self.v is None:
            self.v = {key: 0 for key in grad_dict.keys()}

        self.v = {key: self.rho * v + (1 - self.rho) * g**2 if g is not None else None
                  for (key, v, g) in zip(grad_dict.keys(), self.v.values(), grad_dict.values())}

        return {key: self.learning_rate * g / np.sqrt(v + self.e) if g is not None else None
                for (key, v, g) in zip(grad_dict.keys(), self.v.values(), grad_dict.values())}

    def new_instance(self):
        return RMSprop(self.learning_rate, self.rho, self.e)
