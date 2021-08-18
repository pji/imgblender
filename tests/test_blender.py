"""
test_blender
~~~~~~~~~~~~~~~
"""
import numpy as np

from imgblender import blends
from tests.common import ArrayTestCase, A, B, C, D


# Base test case.
class BlendTestCase(ArrayTestCase):
    def run_test(self, blend, exp, a=A, b=B, *args, **kwargs):
        """Run a basic test on a blend function."""
        # Run test.
        act = blend(a, b, *args, **kwargs)

        # Determine test result.
        self.assertArrayEqual(exp, act, round_=True)


# Test cases.
class ColorBurnTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, divide the value in the base
        image by the value in the blending image.
        """
        exp = np.array([
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.5000, 0.6667, 1.0000, 0.0000],
                [0.0000, 0.6667, 1.0000, 0.6667, 0.0000],
                [0.0000, 1.0000, 0.6667, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
        ], dtype=np.float32)
        blend = blends.color_burn
        self.run_test(blend, exp)


class ColorDodgeTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, increase the value in the base
        image by an amount relative to the value in the blending
        image. The relation is through dividing by the inverse of the
        blending image.
        """
        exp = np.array([
            [
                [0.5000, 0.4286, 0.3333, 0.2000, 0.0000],
                [0.4286, 0.2500, 0.1429, 0.0000, 0.2000],
                [0.3333, 0.1429, 0.0000, 0.1429, 0.3333],
                [0.2000, 0.0000, 0.1429, 0.2500, 0.4286],
                [0.0000, 0.2000, 0.3333, 0.4286, 0.5000],
            ],
        ], dtype=np.float32)
        blend = blends.color_dodge
        self.run_test(blend, exp, C, D)


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
        blend = blends.darker
        self.run_test(blend, exp)


class DifferenceTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, take the absolute value of the
        difference between the two colors.
        """
        exp = np.array([
            [
                [1.0000, 0.5000, 0.0000, 0.5000, 1.0000],
                [0.5000, 0.5000, 0.0000, 0.5000, 0.5000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5000, 0.5000, 0.0000, 0.5000, 0.5000],
                [1.0000, 0.5000, 0.0000, 0.5000, 1.0000],
            ],
        ], dtype=np.float32)
        blend = blends.difference
        self.run_test(blend, exp)


class ExclusionTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, subtract the double product of
        the colors from the sum of the colors.
        """
        exp = np.array([
            [
                [1.0000, 0.6250, 0.5000, 0.6250, 1.0000],
                [0.6250, 0.5000, 0.3750, 0.5000, 0.6250],
                [0.5000, 0.3750, 0.0000, 0.3750, 0.5000],
                [0.6250, 0.5000, 0.3750, 0.5000, 0.6250],
                [1.0000, 0.6250, 0.5000, 0.6250, 1.0000],
            ],
        ], dtype=np.float32)
        blend = blends.exclusion
        self.run_test(blend, exp)


class HardLightTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, perform a hard light blend."""
        exp = np.array([
            [
                [0.0000, 0.3750, 0.5000, 0.6250, 1.0000],
                [0.3750, 1.0000, 0.8750, 1.0000, 0.6250],
                [0.5000, 0.8750, 1.0000, 0.8750, 0.5000],
                [0.6250, 1.0000, 0.8750, 1.0000, 0.3750],
                [1.0000, 0.6250, 0.5000, 0.3750, 0.0000],
            ],
        ], dtype=np.float32)
        blend = blends.hard_light
        self.run_test(blend, exp)


class HardMixTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, perform a hard mix blend."""
        exp = np.array([
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 1.0000, 1.0000, 1.0000, 0.0000],
                [0.0000, 1.0000, 1.0000, 1.0000, 0.0000],
                [0.0000, 1.0000, 1.0000, 1.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
        ], dtype=np.float32)
        blend = blends.hard_mix
        self.run_test(blend, exp)


class LighterTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, always take the highest value."""
        exp = np.array([
            [
                [1.0000, 0.7500, 0.5000, 0.7500, 1.0000],
                [0.7500, 1.0000, 0.7500, 1.0000, 0.7500],
                [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
                [0.7500, 1.0000, 0.7500, 1.0000, 0.7500],
                [1.0000, 0.7500, 0.5000, 0.7500, 1.0000],
            ],
        ], dtype=np.float32)
        blend = blends.lighter
        self.run_test(blend, exp)


class LinearBurnTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, divide the value in the base
        image by the value in the blending image.
        """
        exp = np.array([
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.5000, 0.5000, 0.5000, 0.0000],
                [0.0000, 0.5000, 1.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.5000, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
        ], dtype=np.float32)
        blend = blends.linear_burn
        self.run_test(blend, exp)


class LinearDodgeTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, add the colors together."""
        exp = np.array([
            [
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.5000, 0.2500, 0.2500, 0.2500, 0.5000],
                [0.5000, 0.2500, 0.0000, 0.2500, 0.5000],
                [0.5000, 0.2500, 0.2500, 0.2500, 0.5000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
            ],
        ], dtype=np.float32)
        blend = blends.linear_dodge
        self.run_test(blend, exp, C, D)


class LinearLightTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, perform a linear light blend."""
        exp = np.array([
            [
                [0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
                [0.2500, 1.0000, 1.0000, 1.0000, 0.7500],
                [0.5000, 1.0000, 1.0000, 1.0000, 0.5000],
                [0.7500, 1.0000, 1.0000, 1.0000, 0.2500],
                [1.0000, 0.7500, 0.5000, 0.2500, 0.0000],
            ],
        ], dtype=np.float32)
        blend = blends.linear_light
        self.run_test(blend, exp)


class MultiplyTestCase(BlendTestCase):
    def test_multiply(self):
        """When blending image data, multiply the two values."""
        exp = np.array([
            [
                [0.0000, 0.1875, 0.2500, 0.1875, 0.0000, ],
                [0.1875, 0.5000, 0.5625, 0.5000, 0.1875, ],
                [0.2500, 0.5625, 1.0000, 0.5625, 0.2500, ],
                [0.1875, 0.5000, 0.5625, 0.5000, 0.1875, ],
                [0.0000, 0.1875, 0.2500, 0.1875, 0.0000, ],
            ],
        ], dtype=np.float32)
        blend = blends.multiply
        self.run_test(blend, exp)


class OverlayTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, perform an overlay blend."""
        exp = np.array([
            [
                [0.0000, 0.3750, 0.5000, 0.6250, 1.0000],
                [0.3750, 1.0000, 0.8750, 1.0000, 0.6250],
                [0.5000, 0.8750, 1.0000, 0.8750, 0.5000],
                [0.6250, 1.0000, 0.8750, 1.0000, 0.3750],
                [1.0000, 0.6250, 0.5000, 0.3750, 0.0000],
            ],
        ], dtype=np.float32)
        blend = blends.overlay
        self.run_test(blend, exp)


class PinLightTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, perform an pin light blend."""
        exp = np.array([
            [
                [0.0000, 0.5000, 0.5000, 0.5000, 1.0000],
                [0.5000, 1.0000, 0.7500, 1.0000, 0.5000],
                [0.5000, 0.7500, 1.0000, 0.7500, 0.5000],
                [0.5000, 1.0000, 0.7500, 1.0000, 0.5000],
                [1.0000, 0.5000, 0.5000, 0.5000, 0.0000],
            ],
        ], dtype=np.float32)
        blend = blends.pin_light
        self.run_test(blend, exp)


class ReplaceTestCase(BlendTestCase):
    def test_replace(self):
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
        blend = blends.replace
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


class ScreenTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, increase the value in the base
        image by an amount relative to the value in the blending
        image.
        """
        exp = np.array([
            [
                [1.0000, 0.8125, 0.7500, 0.8125, 1.0000],
                [0.8125, 1.0000, 0.9375, 1.0000, 0.8125],
                [0.7500, 0.9375, 1.0000, 0.9375, 0.7500],
                [0.8125, 1.0000, 0.9375, 1.0000, 0.8125],
                [1.0000, 0.8125, 0.7500, 0.8125, 1.0000],
            ],
        ], dtype=np.float32)
        blend = blends.screen
        self.run_test(blend, exp)


class SoftLightTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, perform an soft light blend."""
        exp = np.array([
            [
                [1.0000, 0.6562, 0.5000, 0.3750, 0.0000],
                [0.6562, 1.0000, 0.8080, 0.7071, 0.3750],
                [0.5000, 0.8080, 1.0000, 0.8080, 0.5000],
                [0.3750, 0.7071, 0.8080, 1.0000, 0.6562],
                [0.0000, 0.3750, 0.5000, 0.6562, 1.0000],
            ],
        ], dtype=np.float32)
        blend = blends.soft_light
        self.run_test(blend, exp)


class VividLightTestCase(BlendTestCase):
    def test_blend(self):
        """When blending image data, perform an vivid light blend."""
        exp = np.array([
            [
                [0.0000, 0.5000, 0.5000, 0.5000, 0.0000],
                [0.5000, 1.0000, 1.0000, 0.0000, 0.5000],
                [0.5000, 1.0000, 0.0000, 1.0000, 0.5000],
                [0.5000, 0.0000, 1.0000, 1.0000, 0.5000],
                [0.0000, 0.5000, 0.5000, 0.5000, 0.0000],
            ],
        ], dtype=np.float32)
        blend = blends.vivid_light
        self.run_test(blend, exp)
