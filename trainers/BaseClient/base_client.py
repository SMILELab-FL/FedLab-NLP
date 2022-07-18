from abc import ABC

import torch

from utils import registry

from fedlab.core.client.trainer import ClientTrainer
from fedlab.core.client.manager import PassiveClientManager
from fedlab.core.client.manager import ORDINARY_TRAINER, SERIAL_TRAINER
from fedlab.utils import MessageCode
from fedlab.core.client.serial_trainer import SubsetSerialTrainer


class BaseClientTrainer(ClientTrainer, ABC):
    def __init__(self, model, client_num=1):

        self.client_num = client_num  # default is 1.
        self.type = SERIAL_TRAINER  # represent serial trainer

        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.device = config.training_config.device
        self._model = model.to(self.device)
        self.rank = config.federated_config.rank
        self.param_list = []
        self.logger = registry.get("logger")

    @property
    def uplink_package(self):
        return self.param_list

    def _train_alone(self, model_parameters, train_loader):
        raise NotImplementedError()

    def _test_alone(self, test_loader, idx):
        raise NotImplementedError()

    def _get_dataloader(self, dataset, client_id):
        """Get :class:`DataLoader` for ``client_id``."""
        raise NotImplementedError()

    def local_process(self, id_list, payload):
        raise NotImplementedError()

    def train(self, model_parameters, id_list):
        """Train model."""
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate quality of local model."""
        raise NotImplementedError()


class BaseClientManager(PassiveClientManager, ABC):
    def __init__(self, network, trainer):
        self.logger = registry.get("logger")
        super().__init__(network, trainer, self.logger)

    def main_loop(self):
        """Actions to perform when receiving a new message, including local trainers.

        Main procedure of each client:
            1. client waits for data from server (PASSIVELY).
            2. after receiving data, client start local model trainers procedure.
            3. client synchronizes with server actively.
        """
        while True:
            sender_rank, message_code, payload = self._network.recv(src=0)

            if message_code == MessageCode.Exit:
                # client exit feedback
                if self._network.rank == self._network.world_size - 1:
                    self._network.send(message_code=MessageCode.Exit, dst=0)
                break

            elif message_code == MessageCode.ParameterUpdate:

                id_list, payload = payload[0].to(
                    torch.int32).tolist(), payload[1:]

                # check the trainer type
                if self._trainer.type == SERIAL_TRAINER:  # serial
                    self._trainer.local_process(
                        id_list=id_list,
                        payload=payload
                    )

                elif self._trainer.type == ORDINARY_TRAINER:  # ordinary
                    assert len(id_list) == 1
                    self._trainer.local_process(payload=payload)

                self.synchronize()

            else:
                raise ValueError(
                    "Invalid MessageCode {}. Please check MessageCode list.".
                        format(message_code))

    def synchronize(self):
        """Synchronize with server"""
        self.logger.info("Uploading information to server.")
        self._network.send(
            content=self._trainer.uplink_package,
            message_code=MessageCode.ParameterUpdate,
            dst=0
        )
