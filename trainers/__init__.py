"""FedLab-NLP's trainers registry in trainer.__init__.py -- IMPORTANT!"""

from trainers.FedBaseTrainer import BaseTrainer
from run.fedavg.trainer import FedAvgTrainer
from run.centralized.trainer import CentralizedTrainer


__all__ = [
    "BaseTrainer",
    "FedAvgTrainer",
    "CentralizedTrainer"
]
