"""
common
~~~~~~

Common utility functions for the imgblender module.
"""
from functools import wraps
from typing import Callable, Union

import numpy as np


# Decorators.
def clipped(fn: Callable) -> Callable:
    """Blends that use division or unbounded addition or
    subtraction can overflow the scale of the image. This will
    keep the image in scale by clipping the values below zero
    to zero and the values above one to one.
    """
    @wraps(fn)
    def wrapper(a: np.ndarray, b: np.ndarray, *args, **kwargs) -> np.ndarray:
        ab = fn(a, b, *args, **kwargs)
        ab[ab < 0.0] = 0.0
        ab[ab > 1.0] = 1.0
        return ab
    return wrapper


def faded(fn: Callable) -> Callable:
    """Adjust how much the operation affects the base array."""
    @wraps(fn)
    def wrapper(a: np.ndarray,
                b: np.ndarray,
                fade: float,
                *args, **kwargs) -> np.ndarray:
        ab = fn(a, b, *args, **kwargs)
        if fade == 1:
            return ab
        ab = a + (ab - a) * fade
        return ab
    return wrapper


def masked(fn: Callable) -> Callable:
    """Apply a blending mask to the image."""
    @wraps(fn)
    def wrapper(a: np.ndarray,
                b: np.ndarray,
                mask: Union[None, np.ndarray] = None,
                *args, **kwargs) -> np.ndarray:
        # Get the blended image from the decorated function.
        ab = fn(a, b, *args, **kwargs)

        # If there wasn't a mask passed in, dont waste time
        # trying to mask the effects.
        if mask is None:
            return ab

        # Apply the mask and return the result.
        ab = a * (1 - mask) + ab * mask
        return ab
    return wrapper
