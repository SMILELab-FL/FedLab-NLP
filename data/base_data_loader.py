from abc import ABC, abstractmethod
from utils import registry, pickle_read
from transformers import AutoTokenizer


class BaseDataLoader(ABC):
    def __init__(self):

        config = registry.get("config")
        self.model_config = config.model_config
        self.data_config = config.data_config
        self.training_config = config.training_config
        self.federated_config = config.federated_config

        self.partition_name = self.federated_config.partition_method
        self.clients_list = self.federated_config.clients_id_list

        self._load_attributes()
        self._build_tokenizer()
        self._build_registry()

        self.logger = registry.get("logger")

    def _load_data(self):
        if self.federated_config.rank == -1:
            self._load_centralized_data()
        elif self.federated_config.rank == 0:
            self._load_federated_data_on_server()
        else:
            self._load_federated_data_on_client()

    def _load_federated_data_on_client(self):
        raise NotImplementedError

    def _load_federated_data_on_server(self):
        raise NotImplementedError

    def _load_centralized_data(self):
        raise NotImplementedError

    def _transformer_data(self, data, data_list):
        raise NotImplementedError

    def _load_cached_data(self):
        raise NotImplementedError

    def _load_attributes(self):
        partition_data = pickle_read(self.data_config.partition_dataset_path)
        self.attribute = partition_data[self.partition_name]["attribute"]

    def _build_registry(self):
        if self.model_config.model_output_mode == "seq_classification":
            registry.register("num_labels", len(self.attribute["label_list"]))

    def _build_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name_or_path)
