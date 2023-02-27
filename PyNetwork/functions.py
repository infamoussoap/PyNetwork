import numpy as np


def get_activation_function(name, **kwargs):
    """ Returns the function of the given name

        Parameters
        ----------
        name : str
            The name of the desired function

        Raises
        ------
        Exception
            If `name` has not been implemented
    """

    if name == 'relu':
        def relu(x, grad=False):
            if grad:
                return (x > 0).astype(int)

            x = np.array(x)
            x[x < 0] = 0
            return x
        # END def relu

        return relu
    elif name == 'softmax':
        def softmax(x, grad=False):
            if grad:
                softmax_val = softmax(x, grad=False)
                return softmax_val*(1 - softmax_val)

            z = x - np.max(x, axis=-1, keepdims=True)
            numerator = np.exp(z)
            denominator = np.sum(numerator, axis=-1, keepdims=True)
            return numerator / denominator
        # END def softmax

        return softmax
    elif name == 'linear':
        def linear(x, grad=False):
            if grad:
                return 1
            return x
        return linear

    else:
        raise Exception(f'{name} is not a defined function.')


def get_error_function(name):
    """ Returns the function of the given name

        Parameters
        ----------
        name : str
            The name of the desired function

        Raises
        ------
        Exception
            If `name` has not been implemented
    """
    if name == 'mse':
        def mse(predictions, targets, grad = False):
            if grad:
                return 2*(predictions - targets)
            N = predictions.shape[0]
            return np.sum(((predictions - targets)**2)/2)/N
        return mse
    elif name == 'cross_entropy':
        def cross_entropy(predictions, targets, epsilon=1e-12, grad=False):
            """ Computes cross entropy between targets (encoded as one-hot vectors) and predictions.

                Parameters
                ----------
                    predictions : (N, k) np.array
                    targets     : (N, k) np.array

                Returns
                -------
                    float
                        If grad = False then the cross_entropy score is retuned

                    OR

                    (N, k) np.array
                        If grad = True then the gradient of the output is returned
            """
            predictions = np.clip(predictions, epsilon, 1. - epsilon)

            if grad:
                return -targets/predictions + (1 - targets)/(1 - predictions)

            N = predictions.shape[0]
            ce = -np.sum(targets*np.log(predictions+1e-9))/N
            return ce
        return cross_entropy
    else:
        raise Exception(f'{name} is not a defined function.')


def get_metric_function(name):
    """ Returns the metric fucntion of a given name

        Parameters
        ----------
        name : str
            The name of the desired function

        Raises
        ------
        Exception
            If `name` has not been implemented
    """
    if name == 'accuracy':
        def accuracy(predictions, target):
            return np.mean(np.argmax(predictions, axis=-1) == np.argmax(target, axis=-1))
        return accuracy
    else:
        raise Exception(f'{name} is not a defined metric.')
