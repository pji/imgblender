"""
blends
~~~~~~

Blending operations to use when combining two sets of image data.

Many of these are taken from:

    *   http://www.deepskycolors.com/archive/2010/04/21/
        formulas-for-Photoshop-blending-modes.html
    *   http://www.simplefilter.de/en/basics/mixmods.html


Basic Usage: Blends
===================
The blending operation functions (blends) are used to blend two sets
of image data together. Using a blending operation (an "operation")
works like any other function all. The parameters follow the Blending
Operation protocol.

Usage::

    >>> import numpy as np
    >>> a = np.array([[[0., .25, .5, .75, 1.], [0., .25, .5, .75, 1.]]])
    >>> b = np.array([[[1., 75, .5, .25, 0.], [1., 75, .5, .25, 0.]]])
    >>> darker(a, b)
    array([[[0.  , 0.25, 0.5 , 0.25, 0.  ],
            [0.  , 0.25, 0.5 , 0.25, 0.  ]]])

While the functions themselves are fairly simple, they are given some
extra functionality by decorators. Ultimately the true protocol for the
operations is:

    :param a: The image data from the existing image.
    :param b: The image data from the blending image.
    :param fade: (Optional.) (From @can_fade.) How much the blend
        should impact the final output. This is a percentage, so the
        range of valid values are 0 <= x <= 1.
    :param mask: (Optional.) (From @mcan_mask.) An array of data used
        to mask the blending operation. This is also a percentage, so a
        value of one in the mask means that pixel is fully affected by
        the operation. A value of zero means the pixel is not affected
        by the operation.
    :return: A :class:numpy.ndarray object.
    :rtype: numpy.ndarray
"""
import numpy as np

from imgblender.common import will_clip, can_fade, can_mask


# Simple replacement blends.
@can_mask
@can_fade
def replace(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Simple replacement filter. Can double as an opacity filter
    if passed can_fade amount, but otherwise this will just replace the
    values in a with the values in b.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    return b


# Darker/burn blends.
@will_clip
@can_mask
@can_fade
def darker(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Replaces values in the existing image with values from the
    blending image when the value in the blending image is darker.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    ab = a.copy()
    ab[b < a] = b[b < a]
    return ab


@will_clip
@can_mask
@can_fade
def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiplies the values of the two images, leading to darker
    values. This is useful for shadows and similar situations.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    return a * b


@will_clip
@can_mask
@can_fade
def color_burn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similar to multiply, but is darker and produces higher
    contrast.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    m = b != 0
    ab = np.zeros_like(a)
    ab[m] = 1 - (1 - a[m]) / b[m]
    ab[~m] = 0
    return ab


@will_clip
@can_mask
@can_fade
def linear_burn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similar to multiply, but is darker, produces less saturated
    colors than color burn, and produces more contrast in the shadows.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    return a + b - 1


# Lighter/dodge blends.
@will_clip
@can_mask
@can_fade
def lighter(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Replaces values in the existing image with values from the
    blending image when the value in the blending image is lighter.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    ab = a.copy()
    ab[b > a] = b[b > a]
    return ab


@will_clip
@can_mask
@can_fade
def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Performs an inverse multiplication on the colors from the two
    images then inverse the colors again. This leads to overall
    brighter colors and is the opposite of multiply.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    rev_a = 1 - a
    rev_b = 1 - b
    ab = rev_a * rev_b
    return 1 - ab


@will_clip
@can_mask
@can_fade
def color_dodge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similar to screen, but brighter and decreases the contrast.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    ab = np.ones_like(a)
    ab[b != 1] = a[b != 1] / (1 - b[b != 1])
    return ab


@will_clip
@can_mask
@can_fade
def linear_dodge(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similar to screen but produces stronger results.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    return a + b


# Inversion blends.
@will_clip
@can_mask
@can_fade
def difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Takes the absolute value of the difference of the two values.
    This is often useful in creating complex patterns or when
    aligning two images.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    return np.abs(a - b)


@will_clip
@can_mask
@can_fade
def exclusion(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similar to difference, with the result tending to gray
    rather than black.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    ab = a + b - 2 * a * b
    return ab


# Contrast blends.
@will_clip
@can_mask
@can_fade
def hard_light(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similar to the blending image being a harsh light shining
    on the existing image.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    ab = np.zeros_like(a)
    ab[a < .5] = 2 * a[a < .5] * b[a < .5]
    ab[a >= .5] = 1 - 2 * (1 - a[a >= .5]) * (1 - b[a >= .5])
    return ab


@will_clip
@can_mask
@can_fade
def hard_mix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Increases the saturation and contrast. It's best used with
    masks and can_fade.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    ab = np.zeros_like(a)
    ab[a < 1 - b] = 0
    ab[a > 1 - b] = 1
    return ab


@will_clip
@can_mask
@can_fade
def linear_light(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combines linear dodge and linear burn.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    ab = b + 2 * a - 1
    return ab


@will_clip
@can_mask
@can_fade
def overlay(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combines screen and multiply blends.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    mask = a >= .5
    ab = np.zeros_like(a)
    ab[~mask] = (2 * a[~mask] * b[~mask])
    ab[mask] = (1 - 2 * (1 - a[mask]) * (1 - b[mask]))
    return ab


@will_clip
@can_mask
@can_fade
def pin_light(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combines lighten and darken blends.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    # Build array masks to handle how the algorithm changes.
    m1 = np.zeros(a.shape, bool)
    m1[b < 2 * a - 1] = True
    m2 = np.zeros(a.shape, bool)
    m2[b > 2 * a] = True
    m3 = np.zeros(a.shape, bool)
    m3[~m1] = True
    m3[m2] = False

    # Blend the arrays using the algorithm.
    ab = np.zeros_like(a)
    ab[m1] = 2 * a[m1] - 1
    ab[m2] = 2 * a[m2]
    ab[m3] = b[m3]
    return ab


@will_clip
@can_mask
@can_fade
def soft_light(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similar to overlay, but biases towards the blending value
    rather than the existing value.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    m = np.zeros(a.shape, bool)
    ab = np.zeros_like(a)
    m[a < .5] = True
    ab[m] = (2 * a[m] - 1) * (b[m] - b[m] ** 2) + b[m]
    ab[~m] = (2 * a[~m] - 1) * (np.sqrt(b[~m]) - b[~m]) + b[~m]
    return ab


@will_clip
@can_mask
@can_fade
def vivid_light(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Good for color grading when faded.

    :param a: The existing values. This is like the bottom layer in
        a photo editing tool.
    :param b: The values to blend. This is like the top layer in a
        photo editing tool.
    :param fade: (Optional.) The amount the blended values should
        affect the existing values. This is a float between zero
        and one, where zero is no effect and one is full effect.
        See common.can_fade for more details.
    :param mask: (Optional.) An image mask that is used to determine
        the effect the blend should have on the existing values.
        This is a :class:numpy.ndarray of floats between zero and
        one, where zero is no effect and one is full effect. See
        common.can_mask for details.
    :return: An array that contains the values of the blended arrays.
    :rtype: np.ndarray
    """
    # Create masks to handle the algorithm change and avoid division
    # by zero.
    m1 = np.zeros(a.shape, bool)
    m1[a <= .5] = True
    m1[a == 0] = False
    m2 = np.zeros(a.shape, bool)
    m2[a > .5] = True
    m2[a == 1] = False

    # Use the algorithm to blend the arrays.
    ab = np.zeros_like(a)
    ab[m1] = 1 - (1 - b[m1]) / (2 * a[m1])
    ab[m2] = b[m2] / (2 * (1 - a[m2]))
    return ab


if __name__ == '__main__':
    from imgblender.common import A, B, C, D, print_array
    fn = vivid_light
    ab = fn(A, B)
#     ab = fn(C, D)
    print_array(ab)
