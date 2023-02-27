""" Utilities to help validating
"""

from PyNetwork.exceptions import NotBuiltError


def check_layer(layer):
    if layer.built:
        return None

    msg = f'Layer {repr(layer)} is being accessed without being built/initialised first. ' + \
          'Either make sure to build the Sequential model, or build this layer specifically.'
    raise NotBuiltError(msg)
