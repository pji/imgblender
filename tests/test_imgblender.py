"""
test_imgblender
~~~~~~~~~~~~~~~

Unit tests for the imgblender.imgblender module.
"""
from unittest.mock import patch

import numpy as np

from imgblender import blends
from imgblender import imgblender as ib
from tests.common import ArrayTestCase


# Test cases.
class BlendImagesTestCase(ArrayTestCase):
    @patch('cv2.imwrite')
    def test_blend(self, mock_imwrite):
        """Given the paths to two files, a blend function, and the
        path of a destination file, run the blend on the contents
        of the two files and save the result in the destination file.
        """
        # Expected values.
        exp_a = np.array([
            [0x00, 0x00, 0x00, ],
            [0x00, 0x3f, 0x7f, ],
            [0x00, 0x7f, 0xff, ],
        ], dtype=np.uint8)
        exp_path = 'spam.jpg'

        # Test data and state.
        file_a = 'tests/data/__test_horizontal_grayscale_image.jpg'
        file_b = 'tests/data/__test_vertical_grayscale_image.jpg'
        blend = blends.multiply

        # Run test and extract results.
        ib.blend_images(file_a, file_b, blend, exp_path)
        args = mock_imwrite.call_args.args
        act_a = args[1]
        act_path = args[0]

        # Determine test result.
        self.assertArrayEqual(exp_a, act_a)
        self.assertEqual(exp_path, act_path)
