"""
imgblender
~~~~~~~~~~

Blending operations to use when combining two sets of image data.
"""
import numpy as np

from imgblender.common import clipped, faded, masked


# Simple replacement blends.
@masked
@faded
def replace_(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Simple replacement filter. Can double as an opacity filter
    if passed an amount, but otherwise this will just replace the
    values in a with the values in b.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    return b


# Darker/burn blends.
@masked
@faded
def darker(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Always select the darkest value."""
    ab = a.copy()
    ab[b < a] = b[b < a]
    return ab
