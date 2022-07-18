from typing import Optional
from dataclasses import dataclass, field


@dataclass
class FederatedTrainingArguments:
    fl_algorithm: str = field(
        default="fedavg",
        metadata={"help": "The name of the federated learning algorithm"},
    )
    clients_num: int = field(
        default=100,
        metadata={"help": "The number of participant clients"},
    )
    alpha: Optional[float] = field(
        default=1.0,
        metadata={"help": "Non-IID shift and None denotes IID"},
    )
    partition_method: str = field(
        default=None,
        metadata={"help": "The partition methods"},
    )
    do_mimic: bool = field(
        default=True, metadata={"help": "Important! we only process once data processing in server"}
    )
    ip: str = field(
        default="127.0.0.1"
    )
    port: str = field(
        default="10001"
    )
    rank: int = field(
        default=0, metadata={"help": "-1: centralized, 0: server, >0: client"}
    )
    world_size: int = field(
        default=None, metadata={"help": "The number of sub-server"}
    )
    ethernet: Optional[str] = field(
        default=None, metadata={"help": "not set"}
    )
    rounds: int = field(
        default=100, metadata={"help": "The number of training round"}
    )
    sample: float = field(
        default=0.1, metadata={"help": "The participant ratio in each training round"}
    )

    _clients_num_per_sub_server: int = field(
        init=False, metadata={"help": "The number of clients in different works"}
    )

    def __post_init__(self):
        if self.alpha is None:
            # IID
            self.alpha = "inf"

        if self.partition_method is None:
            self.partition_method = f"clients={self.clients_num}_alpha={self.alpha}"

        if not self.do_mimic:
            print("Please check whether federated device has its own data")

        if self.world_size is None:
            raise ValueError(f"Must set world_size, but find {self.world_size}")
        else:
            if self.clients_num % (self.world_size-1):
                raise ValueError(f"{self.clients_num} % {(self.world_size-1)} != 0")

    @property
    def clients_num_per_sub_server(self):
        return int(self.clients_num / (self.world_size-1))

    @property
    def clients_id_list(self):
        if self.rank == -1:
            return [1]
        elif self.rank == 0:
            return [i for i in range(self.clients_num)]
        else:
            client_id_end = min(self.clients_num, self.rank * self.clients_num_per_sub_server)
            client_id_list = [
                i for i in range((self.rank - 1) * self.clients_num_per_sub_server, client_id_end)
            ]
            return client_id_list

    @property
    def clients_num_in_total(self):
        if self.rank == -1:
            # centralized
            return 1
        else:
            # federated
            return self.clients_num
