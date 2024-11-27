import argparse

from train import train
from inference import generate_and_show_random_images
from config import CONFIG


if __name__ == "__main__":
    # parse argument: train or inference
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="train or inference")
    args = parser.parse_args()

    cfg = CONFIG()

    if args.mode == "train":
        train(cfg)
    elif args.mode == "inference":
        generate_and_show_random_images(cfg)
    else:
        raise ValueError("Invalid mode. Please choose train or inference.")
