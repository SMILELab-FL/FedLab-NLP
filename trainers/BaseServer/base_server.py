import threading
import random
from abc import ABC

import torch

from utils.register import registry

from fedlab.core.server.handler import Aggregators
from fedlab.core.server.handler import ParameterServerBackendHandler
from fedlab.core.server.manager import ServerManager
from fedlab.utils.serialization import SerializationTool
from fedlab.utils import MessageCode
from fedlab.core.coordinator import Coordinator


class BaseSyncServerHandler(ParameterServerBackendHandler, ABC):
    def __init__(self, model, valid_data, trainer=None):

        self.valid_data = valid_data
        self.trainer = trainer

        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.logger = registry.get("logger")

        self.device = config.training_config.device
        self._model = model.to(self.device)

        # basic setting
        self.client_num_in_total = config.federated_config.clients_num
        self.sample_ratio = config.federated_config.sample

        # client buffer
        self.client_buffer_cache = []
        self.cache_cnt = 0

        # stop condition
        self.global_round = config.federated_config.rounds
        self.round = 0

        #  metrics
        self.global_test_best_metric = float("inf") if self.training_config.is_decreased_valid_metric else -float("inf")
        self.metric_log = {}

        # metric line
        self.metric_name = self.trainer.metric.metric_name
        self.metric_line = f"{self.model_config.model_type}_client={self.federated_config.clients_num}_" \
                           f"alpha={self.federated_config.alpha}_ci={self.federated_config.sample}_"

    def stop_condition(self) -> bool:
        return self.round >= self.global_round

    def sample_clients(self):
        selection = random.sample(
            range(self.client_num_in_total),
            self.client_num_per_round
        )
        return selection

    def _update_global_model(self, payload):
        assert len(payload) > 0

        if len(payload) == 1:
            self.client_buffer_cache.append(payload[0].clone())
        else:
            self.client_buffer_cache += payload  # serial trainer

        assert len(self.client_buffer_cache) <= self.client_num_per_round

        if len(self.client_buffer_cache) == self.client_num_per_round:
            model_parameters_list = self.client_buffer_cache
            self.logger.debug(
                f"Model parameters aggregation, number of aggregation elements {len(model_parameters_list)}"
            )

            # use aggregator
            serialized_parameters = Aggregators.fedavg_aggregate(
                model_parameters_list)
            SerializationTool.deserialize_model(self._model, serialized_parameters)
            self.round += 1

            self.test_on_server()

            # reset cache cnt
            self.client_buffer_cache = []

            return True  # return True to end this round.
        else:
            return False

    @property
    def client_num_per_round(self):
        return max(1, int(self.sample_ratio * self.client_num_in_total))

    @property
    def downlink_package(self):
        """Property for manager layer. BaseServer manager will call this property when activates clients."""
        return [self.model_parameters]

    @property
    def if_stop(self):
        """
        class:`NetworkManager` keeps monitoring this attribute,
        and it will stop all related processes and threads when ``True`` returned.
        """
        return self.round >= self.global_round

    def test_on_server(self):
        raise NotImplementedError()


class BaseServerManager(ServerManager):
    """Synchronous communication

    This is the top class in our framework which is mainly responsible for network communication of SERVER!.
    Synchronously communicate with clients following agreements defined in :meth:`main_loop`.

    Args:
        network (DistNetwork): Network configuration and interfaces.
        handler (ParameterServerBackendHandler): Backend calculation handler for parameter server.
    """

    def __init__(self, network, handler):
        super(BaseServerManager, self).__init__(network, handler)

        self.logger = registry.get("logger")

    def setup(self):
        self._network.init_network_connection()

        rank_client_id_map = {}

        for rank in range(1, self._network.world_size):
            _, _, content = self._network.recv(src=rank)
            rank_client_id_map[rank] = content[0].item()
        self.coordinator = Coordinator(rank_client_id_map, mode='GLOBAL')  # mode='GLOBAL'
        if self._handler is not None:
            self._handler.client_num_in_total = self.coordinator.total

    def main_loop(self):

        while self._handler.if_stop is not True:
            activate = threading.Thread(target=self.activate_clients)
            activate.start()

            while True:
                sender_rank, message_code, payload = self._network.recv()

                if message_code == MessageCode.ParameterUpdate:
                    if self._handler._update_global_model(payload):
                        break
                else:
                    raise Exception(
                        "Unexpected message code {}".format(message_code))

    def shutdown(self):
        """Shutdown stage."""
        self.shutdown_clients()
        super().shutdown()

    def activate_clients(self):

        self.logger.info("BaseClient activation procedure")
        clients_this_round = self._handler.sample_clients()
        rank_dict = self.coordinator.map_id_list(clients_this_round)

        self.logger.info("BaseClient id list: {}".format(clients_this_round))

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(
                content=[id_list] + downlink_package,
                message_code=MessageCode.ParameterUpdate,
                dst=rank
            )

    def shutdown_clients(self):
        """Shutdown all clients.

        Send package to each client with :attr:`MessageCode.Exit`.

        Note:
            Communication agreements related: User can overwrite this function to define package
            for exiting information.
        """
        client_list = range(self._handler.client_num_in_total)
        rank_dict = self.coordinator.map_id_list(client_list)

        for rank, values in rank_dict.items():
            downlink_package = self._handler.downlink_package
            id_list = torch.Tensor(values).to(downlink_package[0].dtype)
            self._network.send(content=[id_list] + downlink_package,
                               message_code=MessageCode.Exit,
                               dst=rank)

        # wait for client exit feedback
        _, message_code, _ = self._network.recv(
            src=self._network.world_size - 1
        )
        assert message_code == MessageCode.Exit
