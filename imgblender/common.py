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
    def wrapper(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = fn(a, b)
        ab[ab < 0.0] = 0.0
        ab[ab > 1.0] = 1.0
        return ab
    return wrapper
