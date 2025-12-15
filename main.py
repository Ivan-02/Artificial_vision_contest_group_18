import argparse
from src.behavior import BehaviorAnalyzer
from src.evaluator import HotaEvaluator
from src.tracker import Tracker
from src.data_manager import *
from src.validator import *
import os


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["prepare_dt", "prepare_bh", "val", "track", "eval", "roi"], help="Mode of operation")
    parser.add_argument("--config", type=str, required=False, help="Config file")
    args = parser.parse_args()

    print(os.getcwd())
    cfg = load_config("./configs/config.yaml")

    if not args.config and args.mode != "eval":
        print("Config file not provided")
        exit(1)
    elif args.mode != "eval":
        cfg_mode = load_config(str(args.config))

    if args.mode == "prepare_dt":
        dm = DataManager(cfg)
        dm.prepare_dataset()

    if args.mode == "prepare_bh":
        dm = DataManager(cfg)
        dm.prepare_behavior_gt()

    elif args.mode == "val":
        validator = Validator(cfg, cfg_mode)
        validator.run()

    elif args.mode == "track":
        validator = Tracker(cfg, cfg_mode)
        validator.run()

    elif args.mode == "eval":
        evaluator = HotaEvaluator(cfg)
        evaluator.run()

    elif args.mode == "roi":
        roi = BehaviorAnalyzer(cfg, cfg_mode)
        roi.run()
