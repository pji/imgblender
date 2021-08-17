"""
test_imgblender
~~~~~~~~~~~~~~~
"""
import numpy as np

from imgblender import imgblender as ib
from tests.common import ArrayTestCase


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


# Base test case.
class BlendTestCase(ArrayTestCase):
    def run_test(self, blend, exp, a=A, b=B, *args, **kwargs):
        """Run a basic test on a blend function."""
        # Run test.
        act = blend(a, b, *args, **kwargs)

        # Determine test result.
        self.assertArrayEqual(exp, act)


# Test cases.
class DarkerTestCase(BlendTestCase):
    def test_darker(self):
        """When blending image data, always take the lowest value."""
        exp = np.array([
            [
                [0.00, 0.25, 0.50, 0.25, 0.00, ],
                [0.25, 0.50, 0.75, 0.50, 0.25, ],
                [0.50, 0.75, 1.00, 0.75, 0.50, ],
                [0.25, 0.50, 0.75, 0.50, 0.25, ],
                [0.00, 0.25, 0.50, 0.25, 0.00, ],
            ],
        ], dtype=np.float32)
        blend = ib.darker
        self.run_test(blend, exp)


class ReplaceTestCase(BlendTestCase):
    def test_replace_(self):
        """When passed two sets of image data, return the second set."""
        # Expected value.
        exp = np.array([
            [
                [1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, ],
            ],
        ], dtype=np.float32)

        # Test data and set up.
        blend = ib.replace_
        a = np.array([
            [
                [0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, ],
            ],
        ], dtype=np.float32)
        b = exp.copy()

        # Run test and determine result
        self.run_test(blend, exp, a, b)
