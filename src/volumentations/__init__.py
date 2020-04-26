"""Fast augmentations for 3d data."""
try:
    from importlib.metadata import version, PackageNotFoundError  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from .core.composition import *
from .core.transforms_interface import *
from .core.serialization import *
from .augmentations.transforms import *
