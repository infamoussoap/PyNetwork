""" This will contain any custom exceptions used in the `Sequential` class
"""


class NotBuiltError(ValueError, AttributeError):
    """ Exception class to be raised when a class of type `Layer` has not been
        built/initialised before it has been used for forward/backwards propagation

        This class inherits from both ValueError and AttributeError because that
        is what scikit does, see
        https://github.com/scikit-learn/scikit-learn/blob/b3ea3ed6a09fe774dfc5160a65172b1bacbb2a82/sklearn/exceptions.py#L21
    """
