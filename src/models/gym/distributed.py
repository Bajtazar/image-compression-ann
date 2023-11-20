import torch.distributed as dist
import torch

from torch.cuda import set_device

import os

SINK_URL: str = "env://"


def init_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(
        backend="nccl", init_method=SINK_URL, world_size=world_size, rank=rank
    )
    set_device(local_rank)
    dist.barrier()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def is_main_process():
    return get_rank() == 0


def call_on_master(callback, *args, **kwargs):
    if is_main_process():
        callback(*args, **kwargs)


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def print_on_master(message: str) -> None:
    if is_main_process():
        print(message)
