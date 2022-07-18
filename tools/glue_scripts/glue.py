import os
import pickle
import argparse
from loguru import logger

from transformers import glue_output_modes as output_modes


from utils import make_sure_dirs
from tools.glue_scripts.partition import GlueDataPartition
from tools.glue_scripts.glue_utils import glue_processors as processors


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task", default=None, type=str, required=True,
        help="Task name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
        help="The output directory to save partition or raw data")
    parser.add_argument("--clients_num", default=None, type=int, required=True,
        help="All clients numbers")
    parser.add_argument("--alpha", default=None, type=float,
        help="The label skew degree.")
    parser.add_argument("--overwrite", default=None, type=int,
        help="overwrite")

    args = parser.parse_args()
    return args


def load_glue_examples(args):
    task_name = args.task.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()

    # if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
    # HACK(label indices are swapped in RoBERTa pretrained model)
    # label_list[1], label_list[2] = label_list[2], label_list[1]

    train_examples = processor.get_train_examples(args.data_dir)
    valid_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)

    return train_examples, valid_examples, test_examples, output_mode, label_list


def get_partition_data(examples, num_classes, num_clients, label_vocab, dir_alpha, partition, ):
    targets = [example.label for example in examples]
    clients_partition_data = GlueDataPartition(
        targets=targets, num_classes=num_classes, num_clients=num_clients,
        label_vocab=label_vocab, dir_alpha=dir_alpha, partition=partition, verbose=False
    )
    assert (len(clients_partition_data) == num_clients,
            "The partition function is wrong, please check")

    partition_data = {}
    for idx in range(len(clients_partition_data)):
        client_idxs = clients_partition_data[idx]
        partition_data[idx] = client_idxs
    return partition_data


def convert_glue_to_federated_pkl():
    logger.info("start...")
    args = parser_args()

    args.data_dir = os.path.join(args.data_dir, args.task)
    logger.info(f"data_dir: {args.data_dir}")
    args.output_dir = os.path.join(args.output_dir, "fedglue")
    make_sure_dirs(args.output_dir)
    args.output_data_file = os.path.join(args.output_dir, f"{args.task.lower()}_data.pkl")
    args.output_partition_file = os.path.join(args.output_dir, f"{args.task.lower()}_partition.pkl")
    logger.info(f"output_dir: {args.output_dir}")

    logger.info("reading examples ...")
    train_examples, valid_examples, test_examples, output_mode, label_list \
        = load_glue_examples(args)

    logger.info("partition data ...")
    lable_mapping = {label: idx for idx, label in enumerate(label_list)}
    attribute = {"lable_mapping": lable_mapping, "label_list": label_list,
                 "clients_num": args.clients_num, "alpha": args.alpha,
                 "output_mode": output_mode
                 }
    clients_partition_data = {"train": get_partition_data(
        examples=train_examples, num_classes=len(label_list), num_clients=args.clients_num,
        label_vocab=label_list, dir_alpha=args.alpha, partition="dirichlet"
    ), "valid": get_partition_data(
        examples=valid_examples, num_classes=len(label_list), num_clients=args.clients_num,
        label_vocab=label_list, dir_alpha=args.alpha, partition="dirichlet"
    ), "test": None, "attribute": attribute}

    logger.info("saving data & partition ...")
    if os.path.isfile(args.output_partition_file):
        logger.info("loading partition data ...")
        with open(args.output_partition_file, "rb") as file:
            partition_data = pickle.load(file)
    else:
        partition_data = {}
    logger.info(f"partition data's keys: {partition_data.keys()}")
    if f"clients={args.clients_num}_alpha={args.alpha}" in partition_data and not args.overwrite:
        logger.info(f"clients={args.clients_num}_alpha={args.alpha} exists "
                    f"and overwrite={args.overwrite}, then skip")
    else:
        logger.info(f"writing clients={args.clients_num}_alpha={args.alpha} ...")
        partition_data[f"clients={args.clients_num}_alpha={args.alpha}"] = clients_partition_data
    with open(args.output_partition_file, "wb+") as file:
        pickle.dump(partition_data, file)

    if os.path.isfile(args.output_data_file) and not args.overwrite:
        logger.info(f"{args.output_data_file} exists "
                    f"and overwrite={args.overwrite}, then skip")
    else:
        data = {
            "train": train_examples, "valid": valid_examples, "test": test_examples
        }
        with open(args.output_data_file, "wb+") as file:
            pickle.dump(data, file)

    logger.info("end")


if __name__ == "__main__":
    convert_glue_to_federated_pkl()
