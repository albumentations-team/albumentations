from importlib.metadata import metadata

try:
    _metadata = metadata("albumentations")
    __version__ = _metadata["Version"]
    __author__ = _metadata["Author"]
    __maintainer__ = _metadata["Maintainer"]
except Exception:  # noqa: BLE001
    __version__ = "unknown"
    __author__ = "Vladimir Iglovikov"
    __maintainer__ = "Vladimir Iglovikov"

import os
from contextlib import suppress

from albumentations.check_version import check_for_updates

from .augmentations import *
from .core.composition import *
from .core.serialization import *
from .core.transforms_interface import *

with suppress(ImportError):
    from .pytorch import *

# Perform the version check after all other initializations
if os.getenv("NO_ALBUMENTATIONS_UPDATE", "").lower() not in {"true", "1"}:
    check_for_updates()
