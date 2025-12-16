r"""Score-based Data Assimilation"""

import os

# Skip mcs module import if SDA_SKIP_MCS is set (for training without jax)
if not os.getenv('SDA_SKIP_MCS'):
    try:
        from . import mcs
    except (ImportError, RuntimeError, AttributeError, SystemError):
        # mcs module requires jax, which may not be available or compatible
        mcs = None
else:
    mcs = None

from . import nn
from . import score
from . import utils
