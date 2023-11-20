from functools import lru_cache
from configparser import ConfigParser
from csv import reader, writer
from itertools import islice


DEFAULT_CONFIG_INI_PATH: str = "config.ini"
MODELS_CSV_FILE_PATH: str = "models/models.csv"


@lru_cache(maxsize=None)
def get_config(path: str = DEFAULT_CONFIG_INI_PATH) -> ConfigParser:
    config = ConfigParser()
    config.read(path)
    return config


def model_row_to_dict(row: tuple[str]) -> tuple[int, dict[str, float | str | int]]:
    (
        id_,
        model,
        train_dataset,
        lambda_,
        lr,
        min_stddev,
        epsilon,
        block_size,
        block_overlap,
    ) = row
    return (
        int(id_),
        {
            "model": model,
            "train_dataset": train_dataset,
            "lambda": float(lambda_),
            "learning_rate": float(lr),
            "min_stddev": float(min_stddev),
            "epsilon": float(epsilon),
            "block_size": int(block_size),
            "block_overlap_size": int(block_overlap),
        },
    )


def config_entry_to_dict(config: ConfigParser) -> dict[str, float | str | int]:
    return {
        "model": config["network"]["model"],
        "train_dataset": config["datasets"]["train"],
        "lambda": float(config["network"]["lambda"]),
        "learning_rate": float(config["network"]["learning_rate"]),
        "min_stddev": float(config["network"]["min_stddev"]),
        "epsilon": float(config["network"]["epsilon"]),
        "block_size": int(config["session"]["block_size"]),
        "block_overlap_size": int(config["session"]["block_overlap_size"]),
    }


def is_network_registered(config: ConfigParser) -> int | None:
    config_entry = config_entry_to_dict(config)
    with open(MODELS_CSV_FILE_PATH, "r") as handle:
        handle = reader(handle)
        for row in islice(handle, 1, None):
            id_, entry = model_row_to_dict(row)
            if entry == config_entry:
                return id_
    return None


def emplace_network_id(config: ConfigParser) -> int:
    with open(MODELS_CSV_FILE_PATH, "r") as handle:
        last_id = list(reader(handle))[-1][0]
    last_id = 0 if last_id == "id" else int(last_id) + 1
    with open(MODELS_CSV_FILE_PATH, "a", newline="") as handle:
        writer(handle).writerow(
            [
                last_id,
                config["network"]["model"],
                config["datasets"]["train"],
                config["network"]["lambda"],
                config["network"]["learning_rate"],
                config["network"]["min_stddev"],
                config["network"]["epsilon"],
                config["session"]["block_size"],
                config["session"]["block_overlap_size"],
            ]
        )
    return last_id


def get_network_id(config: ConfigParser) -> int:
    if (entry_id := is_network_registered(config)) is not None:
        return entry_id
    return emplace_network_id(config)


@lru_cache(maxsize=None)
def current_network_path(config_path: str = DEFAULT_CONFIG_INI_PATH) -> str:
    return (
        f"models/training_session_number_{get_network_id(get_config(config_path)):03}"
    )


def update_config() -> None:
    get_config.cache_clear()
    current_network_path.cache_clear()
