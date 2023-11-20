from rich.progress import Progress, TaskID

from gym.distributed import is_main_process

from contextlib import contextmanager
from typing import TypeVar, Iterable, Generator


Tp = TypeVar("Tp")


class ProgressBar:
    class MockTask:
        def __init__(self) -> None:
            pass

        def update(self, *args, **kwargs):
            pass

    class RealTask:
        def __init__(self, task: TaskID, task_manager: Progress):
            self.__manager = task_manager
            self.__task = task

        def update(self, advance):
            self.__manager.update(self.__task, advance=advance)

    def __init__(self) -> None:
        self.__progress = Progress() if is_main_process() else None

    @contextmanager
    def task(self, name: str, total: int):
        if self.__progress:
            task = self.__progress.add_task(name, total=total)
            yield self.RealTask(task, self.__progress)
            self.__progress.remove_task(task)
        else:
            yield self.MockTask()

    def __enter__(self):
        if self.__progress:
            self.__progress = self.__progress.__enter__()
        return self

    def __exit__(self, *args):
        if self.__progress:
            self.__progress.__exit__(*args)

    def track(self, name: str, iterable: Iterable[Tp]) -> Generator[None, Tp, None]:
        if not self.__progress:
            yield from iterable
            return None
        if not hasattr(iterable, "__len__"):
            iterable = list(iterable)
        with self.task(name, len(iterable)) as task:
            for item in iterable:
                yield item
                task.update(1)
        return None
