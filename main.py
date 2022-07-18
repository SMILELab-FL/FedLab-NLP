from utils import setup_logger, build_config, registry, setup_imports


def main():

    setup_imports()
    setup_logger()
    build_config()

    config = registry.get("config")
    logger = registry.get("logger")

    trainer = registry.get_fl_class(config.federated_config.fl_algorithm)()
    trainer.train()

    logger.info(f"Training Processing is done and "
                f"watch training logs in {trainer.training_config.metric_log_file} and "
                f"training metric in {trainer.training_config.metric_file}")


if __name__ == "__main__":
    main()
