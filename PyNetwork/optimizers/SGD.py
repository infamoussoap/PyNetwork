from PyNetwork.optimizers import Optimizer


class SGD(Optimizer):
    """ Stochastic Gradient Descent Scheduler

        Attributes
        ----------
        learning_rate : (positive) float
            The learning rate or step size
        momentum : (positive) float
            Accelerates gradient descent in the relevant direction
        nesterov : bool
            Whether to apply nesterov momentum
    """

    def __init__(self, learning_rate=0.001, momentum=0.0, nesterov=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov

        self.velocity = None

    @property
    def _gradient_update(self):
        def f(velocity, grad):
            if grad is None:
                return None
            elif self.momentum < 1e-10:
                return self.learning_rate * grad
            elif not self.nesterov:
                return -velocity
            else:
                return -self.momentum * velocity + self.learning_rate * grad

        return f

    def step(self, grad_dict):
        """ Returns the gradients as scheduled by SGD

            Parameters
            ----------
            grad_dict : dict of int - np.array
                Dictionary of gradients, where keys represent the layer number and the
                corresponding value is the layer gradients

            Returns
            -------
            dict of int - np.array
                Dictionary of the gradients as scheduled by SGD. The keys represent the
                layer number, and the corresponding value will be the scheduled gradient

            Notes
            -----
            This function returns the value to subtract from the current parameters.
            Consider grad_dict as dS/da, with a the parameters of the network. Then to
            update the parameters of the network

            a = a - SGD.gradients(dS/da)

            Obviously the output is a dictionary, so you'll have to account for that.
        """

        if self.velocity is None:
            self.velocity = {key: 0 for key in grad_dict.keys()}
        elif self.momentum > 1e-8:
            self.velocity = {key: self.momentum * v - self.learning_rate * g if g is not None else None
                             for (key, v, g) in zip(grad_dict.keys(), self.velocity.values(), grad_dict.values())}

        gradient_updates = {}
        for key, grad in grad_dict.items():
            gradient_updates[key] = self._gradient_update(self.velocity[key], grad)

        return gradient_updates

    def new_instance(self):
        return SGD(self.learning_rate, self.momentum, self.nesterov)
