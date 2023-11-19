from sys import argv
from functools import cache, wraps
from logging import Logger, getLogger, basicConfig, INFO
from typing import Callable, Any, TypeVar
from contextlib import redirect_stdout
from io import StringIO


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


def redirect_stdout_to_logger(level: int = INFO):
    def internal_wrapper(function: Callable[[...], Any]) -> Callable[[...], Any]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with redirect_stdout(StringIO()) as handle:
                try:
                    value = function(*args, **kwargs)
                finally:
                    get_current_logger().log(level, handle.getvalue())
            return value

        return wrapper

    return internal_wrapper
