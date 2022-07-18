from abc import ABC, abstractmethod
from utils import registry
import torch.nn as nn


class BaseModels(nn.Module):
    def __init__(self, task_name):
        super().__init__()

        self.task_name = task_name

        config = registry.get("config")
        self.model_config = config.model_config

    def _build_autoconfig(self):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def forward(self, inputs):
        raise NotImplementedError
