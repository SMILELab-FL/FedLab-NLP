from abc import ABC

from trainers.BaseServer import BaseSyncServerHandler, BaseServerManager


class FedAvgSyncServerHandler(BaseSyncServerHandler, ABC):
    def __init__(self, model, valid_data, trainer=None):
        super().__init__(model, valid_data, trainer=trainer)

        self.metric_log = {
            "model_type": self.model_config.model_type,
            "clients_num": self.federated_config.clients_num,
            "alpha": self.federated_config.alpha, "task": self.data_config.task_name,
            "fl_algorithm": self.federated_config.fl_algorithm,
            "logs": []
        }

    def test_on_server(self):
        # TODO: hard code no metric
        result = self.trainer.eval_model(
            model=self._model,
            valid_dl=self.valid_data
        )

        test_metric, test_loss = result[self.metric_name], result["eval_loss"]
        if self.global_test_best_metric < test_metric:
            self.global_test_best_metric = test_metric
            self.metric_line += f"{self.metric_name}={self.global_test_best_metric:.3f}"

        self.logger.info(f"{self.data_config.task_name}-{self.model_config.model_type} "
                         f"train with client={self.federated_config.clients_num}_"
                         f"alpha={self.federated_config.alpha}_"
                         f"epoch={self.training_config.num_train_epochs}_"
                         f"seed={self.training_config.seed}_"
                         f"comm_round={self.federated_config.rounds}")

        self.logger.debug(f"{self.federated_config.fl_algorithm} Testing "
                          f"Round: {self.round}, Current {self.metric_name}: {test_metric:.3f}, "
                          f"Current Loss: {test_loss:.3f}, Best {self.metric_name}: {self.global_test_best_metric:.3f}")

        self.metric_log["logs"].append(
            {f"round_{self.round}": {
                "loss": f"{test_loss:.3f}",
                f"{self.trainer.metric.metric_name}": f"{self.global_test_best_metric}:.3f"
            }
            }
        )


class FedAvgServerManager(BaseServerManager, ABC):
    def __init__(self, network, handler):
        super().__init__(network, handler)
