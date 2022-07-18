from utils import registry, pickle_write, file_write
from trainers.base_fed_trainer import BaseTrainer
from trainers.fedavg.client import FedAvgClientTrainer, LocalTrainer, FedAvgClientManager
from trainers.fedavg.server import FedAvgSyncServerHandler, FedAvgServerManager


@registry.register_fl_algorithm("fedavg")
class FedAvgTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

        self._before_training()

    def _build_local_trainer(self):
        self.local_trainer = LocalTrainer()

    def _build_server(self):
        self.handler = FedAvgSyncServerHandler(
            self.model, trainer=self.local_trainer,
            valid_data=self.data.test_dataloader,
        )

        self.server_manger = FedAvgServerManager(
            network=self.network,
            handler=self.handler,
        )

    def _build_client(self):

        self.client_trainer = FedAvgClientTrainer(
            model=self.model,
            train_dataset=self.data.train_dataloader_dict,
            valid_dataset=self.data.train_dataloader_dict,
            data_slices=self.federated_config.clients_id_list,
        )

        self.client_manager = FedAvgClientManager(
            trainer=self.client_trainer,
            network=self.network
        )

    def train(self):
        if self.federated_config.rank == 0:
            self.logger.debug(f"Server Start ...")
            self.server_manger.run()
            pickle_write(self.handler.metric_log, self.training_config.metric_log_file)
            file_write(self.handler.metric_line, self.training_config.metric_file, "w+")

        elif self.federated_config.rank > 0:
            self.logger.debug(f"Sub-Server {self.federated_config.rank} Training Start ...")
            self.client_manager.run()
        else:
            self.logger.critical(f"FedAvg's rank meets >= 0, but we get {self.federated_config.rank}")
            return



