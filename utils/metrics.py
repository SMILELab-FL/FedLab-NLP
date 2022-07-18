from abc import ABC

from utils.register import registry
from tools.glue_scripts.glue_metric import glue_compute_metrics


class BaseMetric(ABC):
    def __init__(self, task_name, is_decreased_valid_metric=False):
        super().__init__()

        self.task_name = task_name
        self.is_decreased_valid_metric = is_decreased_valid_metric
        self.best_valid_metric = float("inf") if self.is_decreased_valid_metric else -float("inf")
        self.results = {}

    def update_metrics(self, *args):
        raise NotImplementedError

    @property
    def best_metric(self):
        return self.results

    @property
    def metric_name(self):
        raise NotImplementedError


@registry.register_metric("glue")
class GlueMetric(BaseMetric):
    def __init__(self, task_name, is_decreased_valid_metric=False):
        super().__init__(task_name, is_decreased_valid_metric)

    def update_metrics(self, preds, labels):

        results = glue_compute_metrics(self.task_name, preds, labels)

        cur_valid_metric = results[self.metric_name]
        if self.is_decreased_valid_metric:
            is_best = cur_valid_metric < self.best_valid_metric
        else:
            is_best = cur_valid_metric > self.best_valid_metric

        if is_best:
            self.results.update(results)

    @property
    def metric_name(self):

        glue_metric_name = {
            "cola": "mcc",
            "sst-2": "acc",
            "mrpc": "acc",
            "sts-b": "acc",
            "qqp": "acc",
            "mnli": "acc",
            "mnli-mm": "acc",
            "qnli": "acc",
            "rte": "acc",
            "wnli": "acc"
        }

        return glue_metric_name[self.task_name]
