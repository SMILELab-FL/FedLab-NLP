from abc import ABC

from utils import registry
from models.base_models import BaseModels
from transformers import AutoConfig, AutoModelForSequenceClassification


@registry.register_model("seq_classification")
class SeqClassification(BaseModels, ABC):
    def __init__(self, task_name):
        super().__init__(task_name)

        self.num_labels = registry.get("num_labels")
        self.auto_config = self._build_autoconfig()
        self.backbone = self._build_model()

    def _build_autoconfig(self):
        auto_config = AutoConfig.from_pretrained(
            self.model_config.config_name if self.model_config.config_name else self.model_config.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=self.task_name if self.task_name else None,
            # cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
        )
        return auto_config

    def _build_model(self):
        backbone = AutoModelForSequenceClassification.from_pretrained(
            self.model_config.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_config.model_name_or_path),
            config=self.auto_config,
            # cache_dir=self.model_config.cache_dir,
            revision=self.model_config.model_revision,
            use_auth_token=True if self.model_config.use_auth_token else None,
            # ignore_mismatched_sizes=self.model_config.ignore_mismatched_sizes,
        )
        return backbone

    def forward(self, inputs):
        output = self.backbone(**inputs)
        return output
