from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("atomic-rl")
except PackageNotFoundError:
    __version__ = "0.1.0"

from . import action_selection
from . import buffers
from . import initialization
from . import losses
from . import metrics
from . import network
from . import optimizer
from . import returns
from . import schedules
from . import search
from . import td
from . import utils

__all__ = [
    "action_selection",
    "buffers",
    "initialization",
    "losses",
    "metrics",
    "network",
    "optimizer",
    "returns",
    "schedules",
    "search",
    "td",
    "utils",
]
