import argparse
import yaml
import sys

from train import main as train_main
from evaluate import main as evaluate_main
from extractsynthdata import save_synthetic_datasets

def main():
    parser = argparse.ArgumentParser(description="NILM Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    action = config.get("action")
    if action == "train":
        train_main(config)
    elif action == "evaluate":
        evaluate_main(config)
    elif action == "extractsynthetic":
        extract_config = config.get("extractsynthetic", {})
        save_synthetic_datasets(**extract_config)
    else:
        print("Invalid action specified in config file.")
        sys.exit(1)

if __name__ == "__main__":
    main()
