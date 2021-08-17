"""
test_common
~~~~~~~~~~~
"""
import numpy as np

from imgblender import common as c
from tests.common import ArrayTestCase


# Test cases.
class ClippedTestCase(ArrayTestCase):
    def test_clips(self):
        """When applied to a function, the clipped decorator should
        set any values in the output of the decorated function that
        are greater than one to one and any values that are less
        than zero to zero.
        """
        # Expected value.
        exp = np.array([
            [
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
            ]
        ], dtype=np.float32)

        # Test data and set up.
        a = np.array([
            [
                [-0.5, 0.0, 0.5, 1.0, 1.5, ],
                [-0.5, 0.0, 0.5, 1.0, 1.5, ],
                [-0.5, 0.0, 0.5, 1.0, 1.5, ],
                [-0.5, 0.0, 0.5, 1.0, 1.5, ],
                [-0.5, 0.0, 0.5, 1.0, 1.5, ],
            ],
        ], dtype=np.float32)
        b = np.array([
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
            ],
        ], dtype=np.float32)

        @c.clipped
        def spam(a, b):
            return a + b

        # Run test.
        act = spam(a, b)

        # Determine test result.
        self.assertArrayEqual(exp, act)


class FadedTestCase(ArrayTestCase):
    def test_fades(self):
        """When applied to a function, the faded decorator should
        adjust how much the blending operation affects the base
        image by the given amount.
        """
        # Expected value.
        exp = np.array([
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, ],
                [0.5, 0.5, 0.5, 0.5, 0.5, ],
                [0.5, 0.5, 0.5, 0.5, 0.5, ],
                [0.5, 0.5, 0.5, 0.5, 0.5, ],
                [0.5, 0.5, 0.5, 0.5, 0.5, ],
            ],
        ], dtype=np.float32)

        # Test data and set up.
        a = np.array([
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
            ],
        ], dtype=np.float32)
        b = np.array([
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
            ],
        ], dtype=np.float32)
        fade = 0.5

        @c.faded
        def spam(a, b):
            return b

        # Run test.
        act = spam(a, b, fade)

        # Determine test results.
        self.assertArrayEqual(exp, act)
