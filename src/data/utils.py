from sys import argv
from functools import cache
from logging import Logger, getLogger, basicConfig, INFO

from functools import wraps
from typing import Callable, Any, TypeVar


Tp = TypeVar("Tp")
WrappedFunction = Callable[..., Tp]


def only_once(initializer: Callable[..., None]) -> Callable[WrappedFunction, Tp]:
    initializer()

    def wrapper(function: WrappedFunction) -> WrappedFunction:
        return function

    return wrapper


@cache
def executable_name() -> str:
    return argv[0].split("/")[-1]


def __initialize_logger() -> None:
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    basicConfig(level=INFO, format=log_fmt)


@only_once(__initialize_logger)
def get_current_logger() -> Logger:
    return getLogger(executable_name())
