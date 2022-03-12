import logging

import pkg_resources as pkg

logger = logging.getLogger(__name__)


def check_version(
    current: str = "0.0.0",
    minimum: str = "0.0.0",
    name: str = "version ",
    pinned: bool = False,
    hard: bool = False,
    verbose: bool = False,
):
    """
    Check version vs. required version.
    Adapted from https://github.com/ultralytics/yolov5/blob/c6b4f84/utils/general.py#L293
    """

    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    verbose_info = f"{name}{minimum} required by yolort, but {name}{current} is currently installed"
    if hard:
        assert result, verbose_info  # assert min requirements met
    if verbose and not result:
        logger.warning(verbose_info)
    return result
