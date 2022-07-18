from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments


@dataclass
class TrainArguments(TrainingArguments):

    do_reuse: bool = field(
        default=False, metadata={"help": "whether to load last checkpoint"}
    )
    metric_name: str = field(
        default="glue", metadata={"help": "whether to load last checkpoint"}
    )
    loss_name: str = field(
        default="xent"
    )
    is_decreased_valid_metric: bool = field(
        default=False
    )
