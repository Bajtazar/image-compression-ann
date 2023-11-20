import torch.distributed as dist
from torch import save
from torch.cuda import set_device

from typing import Any, Callable
import os

SINK_URL: str = "env://"


def init_distributed() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl", init_method=SINK_URL, world_size=world_size, rank=rank
    )
    set_device(local_rank)
    dist.barrier()


def is_dist_avail_and_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def is_main_process() -> bool:
    return get_rank() == 0


def call_on_master(callback: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    if is_main_process():
        callback(*args, **kwargs)


def save_on_master(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        save(*args, **kwargs)


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def print_on_master(message: str) -> None:
    if is_main_process():
        print(message)
