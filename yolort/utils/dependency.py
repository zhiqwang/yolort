import importlib.util
import logging
import warnings
from functools import wraps
from typing import Optional

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


# Via: https://github.com/pytorch/audio/blob/main/torchaudio/_internal/module_utils.py
def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


# Via: https://github.com/pytorch/audio/blob/main/torchaudio/_internal/module_utils.py
def requires_module(*modules: str):
    """Decorate function to give error message if invoked without required optional modules.
    This decorator is to give better error message to users rather
    than raising ``NameError:  name 'module' is not defined`` at random places.
    """
    missing = [m for m in modules if not is_module_available(m)]

    if not missing:
        # fall through. If all the modules are available, no need to decorate
        def decorator(func):
            return func

    else:
        req = f"module: {missing[0]}" if len(missing) == 1 else f"modules: {missing}"

        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(f"{func.__module__}.{func.__name__} requires {req}")

            return wrapped

    return decorator


# Via: https://github.com/pytorch/audio/blob/main/torchaudio/_internal/module_utils.py
def deprecated(direction: str, version: Optional[str] = None):
    """Decorator to add deprecation message
    Args:
        direction (str): Migration steps to be given to users.
        version (str or int): The version when the object will be removed
    """

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            message = (
                f"{func.__module__}.{func.__name__} has been deprecated "
                f'and will be removed from {"future" if version is None else version} release. '
                f"{direction}"
            )
            warnings.warn(message, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped

    return decorator
