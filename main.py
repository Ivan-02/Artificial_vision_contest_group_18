import argparse
from src.behavior import BehaviorAnalyzer
from src.contest_runner import ContestRunner
from src.evaluator import Evaluator
from src.tracker import Tracker
from src.data_manager import *
from src.validator import *
import os


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["prepare", "val", "track", "eval_hota", "eval_roi" ,"roi", "contest"], help="Mode of operation")
    parser.add_argument("--config", type=str, required=False, help="Config file")
    args = parser.parse_args()

    print(os.getcwd())
    cfg = load_config("./configs/config.yaml")

    no_config_modes = ["eval_hota", "prepare", "eval_roi"]

    if not args.config and args.mode not in no_config_modes:
        print("Config file not provided")
        exit(1)
    elif args.mode not in no_config_modes:
        cfg_mode = load_config(str(args.config))

    if args.mode == "eval_roi":
        evaluator = Evaluator(cfg)
        evaluator.run_behavior()

    elif args.mode == "contest":
        contest = ContestRunner(cfg, cfg_mode)
        contest.run()

    elif args.mode == "prepare":
        dm = DataManager(cfg)
        dm.prepare_dataset()

    elif args.mode == "val":
        validator = Validator(cfg, cfg_mode)
        validator.run()

    elif args.mode == "track":
        validator = Tracker(cfg, cfg_mode)
        validator.run()

    elif args.mode == "eval_hota":
        evaluator = Evaluator(cfg)
        evaluator.run_hota()

    elif args.mode == "roi":
        roi = BehaviorAnalyzer(cfg, cfg_mode)
        roi.run()
