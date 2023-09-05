# Copyright (c) 2021, Zhiqiang Wang. All Rights Reserved.

from yolort import utils, v5

__version__ = "0.3.0"

try:
    from .version import __version__  # noqa: F401
except ImportError:
    pass
