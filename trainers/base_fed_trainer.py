from abc import ABC
from utils import registry
from utils import setup_seed
from fedlab.core.network import DistNetwork


class BaseTrainer(ABC):
    def __init__(self):

        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.logger = registry.get("logger")

        # self._before_training()

    @property
    def role(self):
        if self.federated_config.rank == 0:
            return "server"
        elif self.federated_config.rank > 0:
            return f"client_{self.federated_config.rank}"
        else:
            return "centralized"

    def _build_server(self):
        raise NotImplementedError

    def _build_client(self):
        raise NotImplementedError

    def _build_local_trainer(self):
        raise NotImplementedError

    def _build_network(self):
        network = DistNetwork(
            address=(self.federated_config.ip, self.federated_config.port),
            world_size=self.federated_config.world_size,
            rank=self.federated_config.rank,
            ethernet=self.federated_config.ethernet)
        return network

    def _build_data(self):
        self.data = registry.get_data_class(self.data_config.dataset_name)()

    def _build_model(self):
        self.model = registry.get_model_class(self.model_config.model_output_mode)(
            task_name=self.data_config.task_name
        )

    def _before_training(self):

        setup_seed(self.training_config.seed)

        self.logger.info(f"{self.role} building dataset ...")
        # set before build model
        self._build_data()

        self.logger.info(f"{self.role} building model ...")
        self._build_model()

        self.logger.info(f"{self.role} building local trainer ...")
        self._build_local_trainer()

        if self.federated_config.rank != -1:
            self.logger.info(f"{self.role} building network ...")
            self.network = self._build_network()

        if self.federated_config.rank == 0:
            self.logger.info("building server ...")
            self._build_server()
        elif self.federated_config.rank > 0:
            self.logger.info(f"building client {self.federated_config.rank} ...")
            self._build_client()

            self.logger.info(f"local rank {self.federated_config.rank}'s client ids "
                             f"is {list(self.data.train_dataloader_dict.keys())}")
        else:
            self.logger.info("building centralized training")
            self._build_local_trainer()

    def train(self):
        raise NotImplementedError
