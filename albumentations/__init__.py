__version__ = "1.4.8"

import os

from albumentations.check_version import check_for_updates

from .augmentations import *
from .core.composition import *
from .core.serialization import *
from .core.transforms_interface import *

# Perform the version check after all other initializations
if os.getenv("NO_ALBUMENTATIONS_UPDATE", "").lower() not in {"true", "1"}:
    check_for_updates()
    check_for_updates()
