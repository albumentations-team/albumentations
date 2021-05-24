from __future__ import absolute_import

__version__ = "0.5.2"

from .core.composition import *
from .core.transforms_interface import *
from .core.serialization import *
from .augmentations import *

try:
    from .imgaug.transforms import *
except ImportError:
    # ImgAug is not installed by default, so we ignore this error.
    # Run `pip install -U albumentations[imgaug] if you need augmentations from imgaug.`
    pass
