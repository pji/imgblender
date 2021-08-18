"""
common
~~~~~~

Common code used in multiple test modules
"""
import unittest as ut

import numpy as np


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


# Base test cases.
class ArrayTestCase(ut.TestCase):
    def assertArrayEqual(self, a, b, round_=False):
        """Assert that two numpy.ndarrays are equal."""
        if round_:
            a = np.around(a, 4)
            b = np.around(b, 4)
        a_list = a.tolist()
        b_list = b.tolist()
        self.assertListEqual(a_list, b_list)

    def assertArrayNotEqual(self, a, b, round_=False):
        """Assert that two numpy.ndarrays are not equal."""
        if round_:
            a = np.around(a, 4)
            b = np.around(b, 4)
        a_list = a.tolist()
        b_list = b.tolist()
        self.assertFalse(a_list == b_list)
