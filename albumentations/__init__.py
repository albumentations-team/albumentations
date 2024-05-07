__version__ = "1.4.6"

from albumentations.check_version import check_for_updates

from .augmentations import *
from .core.composition import *
from .core.serialization import *
from .core.transforms_interface import *

# Perform the version check after all other initializations
check_for_updates()
