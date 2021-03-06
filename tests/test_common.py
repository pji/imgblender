"""
test_common
~~~~~~~~~~~
"""
import numpy as np

from imgblender import common as c
from tests.common import ArrayTestCase


# Test cases.
class CanFadeTestCase(ArrayTestCase):
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

        @c.can_fade
        def spam(a, b):
            return b

        # Run test.
        act = spam(a, b, fade)

        # Determine test results.
        self.assertArrayEqual(exp, act)

    def test_no_fade(self):
        """If no fade is passed, the output of the decorated function
        should be returned without being affected by a fade.
        """
        # Expected value.
        exp = np.array([
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
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
        b = exp.copy()

        @c.can_fade
        def spam(a, b):
            return b

        # Run test.
        act = spam(a, b)

        # Determine test results.
        self.assertArrayEqual(exp, act)


class CanMaskTestCase(ArrayTestCase):
    def test_mask(self):
        """When applied to a function, the can_mask decorator should
        adjust how much the blending operation affects each value
        of the base image based on the appropriate value of the given
        mask.
        """
        # Expected value.
        exp = np.array([
            [
                [0.00, 0.00, 0.00, 0.00, 0.00, ],
                [0.25, 0.25, 0.25, 0.25, 0.25, ],
                [0.50, 0.50, 0.50, 0.50, 0.50, ],
                [0.75, 0.75, 0.75, 0.75, 0.75, ],
                [1.00, 1.00, 1.00, 1.00, 1.00, ],
            ],
        ], dtype=np.float32)

        # Test data and set up.
        a = np.array([
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
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
        mask = np.array([
            [
                [1.00, 1.00, 1.00, 1.00, 1.00, ],
                [0.75, 0.75, 0.75, 0.75, 0.75, ],
                [0.50, 0.50, 0.50, 0.50, 0.50, ],
                [0.25, 0.25, 0.25, 0.25, 0.25, ],
                [0.00, 0.00, 0.00, 0.00, 0.00, ],
            ],
        ], dtype=np.float32)

        @c.can_mask
        def spam(a, b):
            return b

        # Run test.
        act = spam(a, b, mask)

        # Determine test results.
        self.assertArrayEqual(exp, act)

    def test_no_mask(self):
        """If no mask is passed, the output of the decorated function
        should not be can_mask.
        """
        # Expected value.
        exp = np.array([
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, ],
            ],
        ], dtype=np.float32)

        # Test data and set up.
        a = np.array([
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
            ],
        ], dtype=np.float32)
        b = exp.copy()

        @c.can_mask
        def spam(a, b):
            return b

        # Run test.
        act = spam(a, b)

        # Determine test results.
        self.assertArrayEqual(exp, act)


class WillClipTestCase(ArrayTestCase):
    def test_clips(self):
        """When applied to a function, the will_clip decorator should
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

        @c.will_clip
        def spam(a, b):
            return a + b

        # Run test.
        act = spam(a, b)

        # Determine test result.
        self.assertArrayEqual(exp, act)


class WillColorizeTestCase(ArrayTestCase):
    def test_colorize_a(self):
        """Given an RGB image and a grayscale image, the grayscale
        image should be converted to RGB.
        """
        # Expected value.
        exp = np.array([
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
            ]
        ], dtype=np.float32)

        # Test data and set up.
        a = np.array([
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
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

        @c.will_colorize
        def spam(a, b):
            return b

        # Run test.
        act = spam(a, b)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_colorize_b(self):
        """Given an RGB image and a grayscale image, the grayscale
        image should be converted to RGB.
        """
        # Expected value.
        exp = np.array([
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
            ]
        ], dtype=np.float32)

        # Test data and set up.
        a = np.array([
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
            ],
        ], dtype=np.float32)
        b = np.array([
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
            ],
        ], dtype=np.float32)

        @c.will_colorize
        def spam(a, b):
            return a

        # Run test.
        act = spam(a, b)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_no_effect_when_both_grayscale(self):
        """If both images only have one channel, the decorator doesn't
        change either array.
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
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
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

        @c.will_colorize
        def spam(a, b):
            return a + b

        # Run test.
        act = spam(a, b)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_no_effect_when_both_rgb(self):
        """If both images have three channels, the decorator doesn't
        change either array.
        """
        # Expected value.
        exp = np.array([
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
            ]
        ], dtype=np.float32)

        # Test data and set up.
        a = np.array([
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
            ],
        ], dtype=np.float32)
        b = np.array([
            [
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], ],
            ]
        ], dtype=np.float32)

        @c.will_colorize
        def spam(a, b):
            return a + b

        # Run test.
        act = spam(a, b)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_no_effect_when_off(self):
        """When False is passed to the colorize parameter, the
        decorator should not change the image data.
        """
        # Expected value.
        exp = np.array([
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
                [1.0, 1.0, 1.0, 1.0, 1.0, ],
            ],
        ], dtype=np.float32)

        # Test data and set up.
        a = np.array([
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ],
            ],
        ], dtype=np.float32)
        b = exp.copy()
        colorize = False

        @c.will_colorize
        def spam(a, b):
            return b

        # Run test.
        act = spam(a, b, colorize=colorize)

        # Determine test result.
        self.assertArrayEqual(exp, act)


class WillMatchSizeTestCase(ArrayTestCase):
    def test_clips(self):
        """When applied to a function, the will_match_size decorator
        should increase the size of a smaller image to the size of
        the larger image. The fill for the added area should be black.
        """
        # Expected value.
        exp = np.array([
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
            ]
        ], dtype=np.float32)

        # Test data and set up.
        a = np.array([
            [
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
                [0.0, 0.0, 0.5, 1.0, 1.0, ],
            ],
        ], dtype=np.float32)
        b = np.array([
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ],
            ],
        ], dtype=np.float32)

        @c.will_match_size
        def spam(a, b):
            return a + b

        # Run test.
        act = spam(a, b)

        # Determine test result.
        self.assertArrayEqual(exp, act)
