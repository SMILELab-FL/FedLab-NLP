import os
import time
from abc import ABC
# from omegaconf import OmegaConf
from transformers import HfArgumentParser

from utils import make_sure_dirs, rm_file
from utils.register import registry
from configs import ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments


class Config(ABC):
    def __init__(self, model_args, data_args, training_args, federated_args):
        self.model_config = model_args
        self.data_config = data_args
        self.training_config = training_args
        self.federated_config = federated_args

    def save_configs(self):
        pass


def amend_config(model_args, data_args, training_args, federated_args):
    # set training path
    training_args.output_dir = os.path.join(training_args.output_dir, data_args.task_name)
    make_sure_dirs(training_args.output_dir)

    if not data_args.cache_dir:
        cache_dir = os.path.join(training_args.output_dir, "cached_data")
        data_args.cache_dir = os.path.join(
            cache_dir, f"cached_{model_args.model_type}_{federated_args.clients_num}_{federated_args.alpha}"
        )
    make_sure_dirs(data_args.cache_dir)

    training_args.save_dir = os.path.join(training_args.output_dir, federated_args.fl_algorithm.lower())
    make_sure_dirs(training_args.save_dir)
    training_args.checkpoint_dir = os.path.join(training_args.save_dir, "saved_model")
    make_sure_dirs(training_args.checkpoint_dir)

    if federated_args.do_mimic and federated_args.rank == 0:
        server_write_flag_path = os.path.join(data_args.cache_dir, "server_write.flag")
        rm_file(server_write_flag_path)

    # set phase
    phase = "train" if training_args.do_train else "evaluate"
    registry.register("phase", phase)

    # set metric log path
    times = time.strftime("%Y%m%d%H%M%S", time.localtime())
    training_args.metric_file = os.path.join(training_args.save_dir, f"{model_args.model_type}.eval")
    training_args.metric_log_file = os.path.join(training_args.save_dir, f"{times}_{model_args.model_type}.eval.log")

    return model_args, data_args, training_args, federated_args


def build_config():
    # read parameters
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainArguments, FederatedTrainingArguments))
    model_args, data_args, training_args, federated_args = parser.parse_args_into_dataclasses()

    model_args, data_args, training_args, federated_args = \
        amend_config(model_args, data_args, training_args, federated_args)

    # register configs
    config = Config(model_args, data_args, training_args, federated_args)
    registry.register("config", config)

    logger = registry.get("logger")
    # logging import parameters
    logger.critical(f"FL-Algorithm: {config.federated_config.fl_algorithm}")

    # logging some path
    logger.info(f"output_dir: {config.training_config.output_dir}")
    logger.info(f"cache_dir: {config.data_config.cache_dir}")
    logger.info(f"save_dir: {config.training_config.save_dir}")
    logger.info(f"checkpoint_dir: {config.training_config.checkpoint_dir}")
