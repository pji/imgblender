"""
common
~~~~~~

Common utility functions for the imgblender module.
"""
from functools import wraps
from typing import Callable, Union

import numpy as np


# Global data.
X, Y, Z = 2, 1, 0


# Decorators.
def can_fade(fn: Callable) -> Callable:
    """Adjust how much the blend affects the base array."""
    @wraps(fn)
    def wrapper(a: np.ndarray,
                b: np.ndarray,
                fade: float = 1.0,
                *args, **kwargs) -> np.ndarray:
        # Get the blended image from the masked function.
        ab = fn(a, b, *args, **kwargs)

        # If the fade wouldn't change the blended image, don't waste
        # time trying to calculate the effect.
        if fade == 1.0:
            return ab

        # Apply the fade and return the result.
        ab = a + (ab - a) * fade
        return ab
    return wrapper


def can_mask(fn: Callable) -> Callable:
    """Apply a blending mask to the image."""
    @wraps(fn)
    def wrapper(a: np.ndarray,
                b: np.ndarray,
                mask: Union[None, np.ndarray] = None,
                *args, **kwargs) -> np.ndarray:
        # Get the blended image from the decorated function.
        ab = fn(a, b, *args, **kwargs)

        # If there wasn't a mask passed in, don't waste time
        # trying to mask the effects.
        if mask is None:
            return ab

        # Apply the mask and return the result.
        ab = a * (1 - mask) + ab * mask
        return ab
    return wrapper


def will_clip(fn: Callable) -> Callable:
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


def will_colorize(fn: Callable) -> Callable:
    """Ensure the images have the same number of color
    channels.
    """
    @wraps(fn)
    def wrapper(a: np.ndarray,
                b: np.ndarray,
                colorize: bool = True,
                *args, **kwargs) -> np.ndarray:
        # If the image have different numbers of color channels,
        # add color channels to the one with the fewest.
        if colorize:
            a_dims = len(a.shape)
            b_dims = len(b.shape)
            if a_dims + 1 == b_dims and b.shape[-1] == 3:
                a = _grayscale_to_rgb(a)
            elif b_dims + 1 == a_dims and a.shape[-1] == 3:
                b = _grayscale_to_rgb(b)

        # Blend and return.
        ab = fn(a, b, *args, **kwargs)
        return ab
    return wrapper


def will_match_size(fn: Callable) -> Callable:
    """If the given images are different sizes, increase the size of
    the smaller image to match the larger image. Since this affects
    the size of the images, this will need to go before any decorators
    that use the original images to affect the resulting image.
    """
    @wraps(fn)
    def wrapper(a: np.ndarray, b: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Calculate the new size of the images.
        size = tuple(max(dim) for dim in zip(a.shape, b.shape))

        # Resize the dimensions of the arrays that are smaller than
        # the new array size.
        a = _resize_array(a, size)
        b = _resize_array(b, size)

        # Blend and return.
        ab = fn(a, b, *args, **kwargs)
        return ab
    return wrapper


# Debugging utilities.
def print_array(a: np.ndarray, depth: int = 0, color: bool = True) -> None:
    """Write the values of the given array to stdout."""
    if len(a.shape) > 1:
        print(' ' * (4 * depth) + '[')
        for i in range(a.shape[0]):
            print_array(a[i], depth + 1, color)
        print(' ' * (4 * depth) + '],')

    else:
        if a.dtype == np.float32 or a.dtype == np.float64:
            tmp = '{:>1.4f}'
        else:
            tmp = '{}'
        nums = [tmp.format(n) for n in a]
        print(' ' * (4 * depth) + '[' + ', '.join(nums) + '],')


# Private utility functions.
def _grayscale_to_rgb(a: np.ndarray) -> np.ndarray:
    """Convert single channel image data to three channel."""
    new_shape = (*a.shape, 3)
    new_a = np.zeros(new_shape, dtype=a.dtype)
    for channel in range(3):
        new_a[..., channel] = a
    return new_a


def _resize_array(a: np.ndarray,
                  size: tuple[int, ...],
                  fill: float = 0.0) -> np.ndarray:
    """Resize the array to the given size."""
    # Create array at the new size.
    resized = np.full(size, fill, dtype=a.dtype)

    # Determine the amount the image has to be inset by in each dimension.
    size_diff = [n - o for n, o in zip(size, a.shape)]
    pad = [dim // 2 for dim in size_diff]
    end = [n + o for n, o in zip(pad, a.shape)]

    # Place the image and return.
    resized[pad[Z]:end[Z], pad[Y]:end[Y], pad[X]:end[X]] = a
    return resized


# Common sample data.
A = np.array([
    [
        [0.00, 0.25, 0.50, 0.75, 1.00, ],
        [0.25, 0.50, 0.75, 1.00, 0.75, ],
        [0.50, 0.75, 1.00, 0.75, 0.50, ],
        [0.75, 1.00, 0.75, 0.50, 0.25, ],
        [1.00, 0.75, 0.50, 0.25, 0.00, ],
    ],
], dtype=np.float32)
B = np.array([
    [
        [1.00, 0.75, 0.50, 0.25, 0.00, ],
        [0.75, 1.00, 0.75, 0.50, 0.25, ],
        [0.50, 0.75, 1.00, 0.75, 0.50, ],
        [0.25, 0.50, 0.75, 1.00, 0.75, ],
        [0.00, 0.25, 0.50, 0.75, 1.00, ],
    ],
], dtype=np.float32)
C = np.array([
    [
        [0.5000, 0.3750, 0.2500, 0.1250, 0.0000, ],
        [0.3750, 0.2500, 0.1250, 0.0000, 0.1250, ],
        [0.2500, 0.1250, 0.0000, 0.1250, 0.2500, ],
        [0.1250, 0.0000, 0.1250, 0.2500, 0.3750, ],
        [0.0000, 0.1250, 0.2500, 0.3750, 0.5000, ],
    ],
], dtype=np.float32)
D = np.array([
    [
        [0.0000, 0.1250, 0.2500, 0.3750, 0.5000, ],
        [0.1250, 0.0000, 0.1250, 0.2500, 0.3750, ],
        [0.2500, 0.1250, 0.0000, 0.1250, 0.2500, ],
        [0.3750, 0.2500, 0.1250, 0.0000, 0.1250, ],
        [0.5000, 0.3750, 0.2500, 0.1250, 0.0000, ],
    ],
], dtype=np.float32)
