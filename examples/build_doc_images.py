"""
build_doc_images
~~~~~~~~~~~~~~~~

Build the images for the :mod:`imgblender` documentation.
"""
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import imgwriter as iw
import numpy as np

import imgblender as ib


# Constants.
X, Y, Z = -1, -2, -3


# Make the example images.
def make_base_images(size: ib.Size) -> tuple[ib.ImgAry, ib.ImgAry]:
    """Make the base images for the blend."""
    a = np.indices((size[X],), dtype=float)
    a /= size[X] - 1
    a = a.reshape(1, 1, size[X])
    a = np.tile(a, (1, size[Y], 1))

    b = np.indices((size[Y],), dtype=float)
    b /= size[Y] - 1
    b = b.reshape(1, size[Y], 1)
    b = np.tile(b, (1, 1, size[X]))

    return a, b


def make_images(path: Path, size: ib.Size):
    """Make the documentation images."""
    print('Making base images.')
    a, b = make_base_images(size)
    iw.save(path / 'a.jpg', a)
    iw.save(path / 'b.jpg', b)
    print('Base images made.')

    blends = [
        ib.replace,
        ib.darker,
    ]

    for blend in blends:
        fname = f'{blend.__name__}.jpg'
        print(f'Making {fname}...')
        ab = blend(a, b)
        iw.save(path / fname, ab)
        print(f'{fname} made.')


# Mainline.
if __name__ == '__main__':
    p = ArgumentParser(
        description='Generate images for the documentation for imggen.',
        prog='imggen'
    )
    p.add_argument(
        '--outdir', '-o',
        action='store',
        default=Path('docs/source/images'),
        help='The directory to save the images.',
        type=Path
    )
    p.add_argument(
        '--size', '-s',
        action='store',
        default=(1280, 720),
        help='The height and width dimensions of the images.',
        nargs=2,
        type=int
    )
    args = p.parse_args()

    size = (1, args.size[1], args.size[0])
    path = args.outdir
    make_images(path, size)

