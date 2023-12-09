#!/usr/bin/env python3
from gym.gym import Gym
from gym.data_manager import DataManager
from gym.loss_function import NormalDistributionRateDistortionLoss
from gym.quantization import Quantization
from gym.distributed import init_distributed, print_on_master
from gym.config import get_config


def worker(is_distributed: bool) -> None:
    if is_distributed:
        init_distributed()
    config = get_config()
    manager = DataManager(
        test_set_path=config["datasets"]["test"],
        raw_test_set_path=config["datasets"]["raw_test"],
        train_set_path=config["datasets"]["train"],
        batch_size=int(config["environment"]["batch_size"]),
        block_overlap_size=int(config["session"]["block_overlap_size"]),
        train_dataset_split_coeff=float(
            config["environment"]["training_validation_set_split_coeff"]
        ),
        workers=int(config["environment"]["workers"]),
        distributed=is_distributed,
    )
    print_on_master("Loaded samples")
    gym = Gym(
        manager,
        Gym.Params(NormalDistributionRateDistortionLoss(), Quantization()),
    )
    gym.train(
        int(get_config()["environment"]["epochs"]),
        int(get_config()["environment"]["saving_interval"]),
    )
