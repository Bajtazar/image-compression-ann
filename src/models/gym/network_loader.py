from gym.config import get_config, current_network_path
from gym.quantization import Quantization

from torch import load
from torch.optim import Adam

import os


__INITIALIZED: bool = False
NETWORK_DIR_PATH: str = "src/models/networks"


def __load_network() -> None:
    module_path = f"{NETWORK_DIR_PATH}/{get_config()['network']['model']}"
    with open(module_path, "r") as handle:
        exec(handle.read(), globals(), globals())
        if "Network" not in globals():
            raise ImportError(f'"Network" cannot be imported from "{module_path}"')


if not __INITIALIZED:
    __INITIALIZED = True
    __load_network()


def load_network(
    loss_function: callable,
    quantization: Quantization,
    device: str,
    path: str = current_network_path(),
) -> tuple[Network, int]:
    network = Network(
        loss_function=loss_function, device=device, quantization=quantization
    )
    if not os.path.exists(path):
        return network, 0
    network.load_state_dict(load(f"{path}/latest/model"))
    with open(f"{path}/latest/info", "r") as handle:
        epoch = int(list(handle)[0]) + 1
    return network, epoch


def load_optimizer(model: Network, path: str = current_network_path()) -> Adam:
    optimizer = Adam(
        model.parameters(), float(get_config()["network"]["learning_rate"])
    )
    if os.path.exists(path):
        optimizer.load_state_dict(load(f"{path}/latest/optimizer"))
    return optimizer
