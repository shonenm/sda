r"""Score-based Data Assimilation"""

import os

# Skip mcs module import if SDA_SKIP_MCS is set (for training without jax)
if not os.getenv("SDA_SKIP_MCS"):
    try:
        from . import mcs
    except (ImportError, RuntimeError, AttributeError, SystemError):
        # mcs module requires jax, which may not be available or compatible
        mcs = None
else:
    mcs = None

from . import config as config
from . import console as console
from . import data as data
from . import logging as logging
from . import nn as nn
from . import paths as paths
from . import score as score
from . import tracking as tracking
from . import utils as utils

__all__ = ["config", "console", "data", "logging", "mcs", "nn", "paths", "score", "tracking", "utils"]
