from __future__ import absolute_import

__version__ = "1.3.0"

from .augmentations import *
from .core.composition import *
from .core.serialization import *
from .core.transforms_interface import *

try:
    from .imgaug.transforms import *  # type: ignore
except ImportError:
    # imgaug is not installed by default, so we import stubs.
    # Run `pip install -U albumentations[imgaug] if you need augmentations from imgaug.`
    from .imgaug.stubs import *  # type: ignore
