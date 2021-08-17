"""
common
~~~~~~

Common utility functions for the imgblender module.
"""
from functools import wraps
from typing import Callable

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
