from __future__ import annotations
from typing import TypeVar
from dataclasses import dataclass
from os import makedirs
from os.path import exists
from shutil import rmtree
from datetime import datetime
from gc import collect
from os import makedirs

from torch import no_grad
from torch.cuda import empty_cache
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

import numpy as np

from gym.progress_bar import ProgressBar
from gym.data_manager import DataManager
from gym.distributed import save_on_master, is_main_process, get_rank, print_on_master
from gym.network_loader import load_network, current_network_path, load_optimizer, Network
from gym.config import get_config
from gym.quantization import Quantization


Model = TypeVar("Model")


def empty_memory() -> None:
    collect()
    empty_cache()


class Gym:
    @dataclass
    class Params:
        loss_function: callable
        quantization: Quantization
        output_dir: str = current_network_path()

    def __init__(self, data_manager: DataManager, params: Gym.Params) -> None:
        self.__data_manager = data_manager
        self.__params = params
        self.__start_epoch = self.__load_model()
        self.__optimizer = load_optimizer(self.__model)
        self.__snapshot_interval = int(get_config()["environment"]["snapshot_interval"])

    def __load_model(self) -> int:
        self.__model, epoch = load_network(
            loss_function=self.__params.loss_function,
            device=self.__data_manager.platform,
            quantization=self.__params.quantization,
        )
        if self.__data_manager.distributed:
            self.__model = SyncBatchNorm.convert_sync_batchnorm(self.__model.cuda())
            self.__model = DistributedDataParallel(
                self.__model, device_ids=[get_rank()]
            )
        else:
            self.__model = self.__model.cuda()
        return epoch

    @property
    def __module(self) -> Network:
        return self.__model.module if self.__data_manager.distributed else self.__model

    def __train_pass(self, epoch: int) -> None:
        self.__model.train()
        epoch_loss = []
        with self.__task_bar.task(
            f"[green](Epoch={epoch+1}) Training pass status",
            total=self.__data_manager.training_set_len,
        ) as train_tast:
            for batch in self.__data_manager.training_set(epoch):
                latent, hyperlatent, stddev, mean_, recon = self.__model(batch)
                loss = self.__module.loss(
                    latent, hyperlatent, stddev, mean_, recon, batch
                )
                epoch_loss.append(loss.item())
                loss.backward()
                self.__optimizer.step()
                self.__optimizer.zero_grad()
                del batch
                empty_memory()
                train_tast.update(1)
        return np.mean(epoch_loss)

    def __validation_pass(self, epoch: int) -> None:
        self.__model.eval()
        epoch_loss = []
        with self.__task_bar.task(
            f"[green](Epoch={epoch+1}) Validation pass status",
            total=self.__data_manager.validation_set_len,
        ) as val_task:
            with no_grad():
                for batch in self.__data_manager.validation_set(epoch):
                    latent, hyperlatent, stddev, mean_, recon = self.__model(batch)
                    loss = self.__module.loss(
                        latent, hyperlatent, stddev, mean_, recon, batch
                    )
                    epoch_loss.append(loss.item())
                    del batch
                    empty_memory()
                    val_task.update(1)
        return np.mean(epoch_loss)

    def __test_pass(self, epoch: int) -> None:
        self.__model.eval()
        with self.__task_bar.task(
            f"[green](Epoch={epoch+1}) Test pass status",
            total=self.__data_manager.test_set_len,
        ) as test_task:
            with no_grad():
                for batch, origin in self.__data_manager.test_set(epoch):
                    _, _, _, _, output = self.__model(batch)
                    self.__data_manager.save_image(
                        output, f"{self.__params.output_dir}/{epoch}/tmp/{origin}"
                    )
                    del batch
                    empty_memory()
                    test_task.update(1)

    def __reconstruct_images(self, epoch: int) -> None:
        with self.__task_bar.task(
            f"[green](Epoch={epoch+1}) Test images reconstruction",
            total=self.__data_manager.real_test_set_len,
        ) as recon_task:
            if int(get_config()["environment"]["make_snapshot_on_save"]):
                save_on_master(
                    self.__module.state_dict(),
                    f"{self.__params.output_dir}/{epoch}/model",
                )
                save_on_master(
                    self.__optimizer.state_dict(),
                    f"{self.__params.output_dir}/{epoch}/optimizer",
                )
            self.__data_manager.reconstruct_images(
                f"{self.__params.output_dir}/{epoch}/tmp",
                f"{self.__params.output_dir}/{epoch}/result",
                lambda: recon_task.update(1),
            )
            rmtree(f"{self.__params.output_dir}/{epoch}/tmp")

    def __save_model(self, epoch: int) -> None:
        try:
            makedirs(f"{self.__params.output_dir}/{epoch}/tmp")
        except Exception:
            pass
        self.__test_pass(epoch)
        self.__reconstruct_images(epoch)

    def __print_status(
        self, epoch: int, epochs: int, train_loss: float, val_loss: float
    ) -> None:
        time = datetime.now().strftime("%H:%M:%S")
        message = f"[{time}]Epoch: {epoch + 1}/{epochs} ({(((epoch + 1) / epochs) * 100):.2f}%) losses: t={train_loss}, v={val_loss}"
        print_on_master(message)
        log_path = f"{self.__params.output_dir}/loss.csv"
        if not exists(log_path):
            self.__create_csv_header(log_path)
        with open(log_path, "a") as handle:
            handle.write(f"{time},{epoch},{train_loss},{val_loss}\n")

    def __create_csv_header(self, log_path: str) -> None:
        makedirs(self.__params.output_dir, exist_ok=True)
        with open(log_path, "a") as handle:
            handle.write("Time,Epoch,Train loss,Validation loss\n")

    def __save_latest_iteration(self, epoch: int) -> None:
        if not exists(f"{self.__params.output_dir}/latest/"):
            try:
                makedirs(f"{self.__params.output_dir}/latest/")
            except Exception:
                pass
        save_on_master(
            self.__module.state_dict(), f"{self.__params.output_dir}/latest/model"
        )
        save_on_master(
            self.__optimizer.state_dict(),
            f"{self.__params.output_dir}/latest/optimizer",
        )
        with open(f"{self.__params.output_dir}/latest/info", "w") as handle:
            handle.write(str(epoch))

    def __signalize(self, status: str) -> None:
        with open("status", "w") as handle:
            handle.write(status)

    def __save_model_if_ready(self, epoch: int, save_interval: int) -> None:
        if epoch % save_interval == save_interval - 1 or epoch == 0:
            self.__save_model(epoch)
        if epoch % self.__snapshot_interval == self.__snapshot_interval - 1:
            self.__save_latest_iteration(epoch)

    def __train_iteration(
        self, epoch: int, save_interval: int
    ) -> tuple[bool, np.float64, np.float64]:
        train_pass_value = self.__train_pass(epoch)
        val_pass_value = self.__validation_pass(epoch)
        if np.isnan(train_pass_value) or np.isnan(val_pass_value):
            return False, train_pass_value, val_pass_value
        if is_main_process():
            self.__save_model_if_ready(epoch, save_interval)
        return True, train_pass_value, val_pass_value

    def __train_loop(
        self, epochs: int, save_interval: int, main_task: ProgressBar
    ) -> None:
        for epoch in range(self.__start_epoch, epochs):
            is_valid, train_loss, val_loss = self.__train_iteration(
                epoch, save_interval
            )
            self.__print_status(epoch, epochs, train_loss, val_loss)
            if not is_valid:
                self.__signalize("NaN")
                exit(0)
            main_task.update(1)

    def train(self, epochs: int, save_interval: int) -> None:
        with ProgressBar() as progress:
            self.__task_bar = progress
            with self.__task_bar.task(
                "[red]Model training status", total=epochs
            ) as main_task:
                main_task.update(self.__start_epoch)
                self.__train_loop(epochs, save_interval, main_task)
        self.__signalize("Done")
