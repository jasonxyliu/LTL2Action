import os
import sys
import logging
import csv
import pickle
import torch

import utils


def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)


def get_storage_dir():
    if "RL_STORAGE" in os.environ:
        return os.environ["RL_STORAGE"]
    return "storage"


def get_model_dir(model_name, storage_dir="storage"):
    return os.path.join(storage_dir, model_name)


def get_status_path(model_dir):
    return os.path.join(model_dir, "status.pt")


def get_status(model_dir, use_cpu):
    path = get_status_path(model_dir)
    if use_cpu:
        return torch.load(path, map_location=torch.device("cpu"))
    return torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def save_status(status, model_dir):
    path = get_status_path(model_dir)
    utils.create_folders_if_necessary(path)
    torch.save(status, path)


def get_vocab(model_dir):
    return get_status(model_dir)["vocab"]


def get_model_state(model_dir, use_cpu):
    return get_status(model_dir, use_cpu)["model_state"]


def get_txt_logger(model_dir):
    path = os.path.join(model_dir, "log.txt")
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger()


def get_csv_logger(model_dir):
    csv_path = os.path.join(model_dir, "log.csv")
    utils.create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)


def load_config(model_dir):
    path = os.path.join(model_dir, "config.pickle")
    if (not os.path.exists(path)):
        print(f"No config file found at: {path}")

    return pickle.load(open(path, "rb"))


def save_config(model_dir, config):
    path = os.path.join(model_dir, "config.pickle")
    utils.create_folders_if_necessary(path)

    pickle.dump(config, open(path, "wb"))
