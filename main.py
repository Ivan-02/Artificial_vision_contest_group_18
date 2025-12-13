import argparse
import yaml
from src.data_manager import *
from src.validator import *
import os


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["prepare", "train", "val"], help="Mode of operation")
    parser.add_argument("--verbose",action="store_true", help="Increase output verbosity")
    args = parser.parse_args()

    print(os.getcwd())
    cfg = load_config("./configs/config.yaml")
    verbose = args.verbose

    if args.mode == "prepare":
        dm = DataManager(cfg)
        dm.prepare_dataset()

    elif args.mode == "val":
        validator = Validator(cfg)
        validator.run(verbose=verbose)

    elif args.mode == "train":
        pass