#
# Main file
#
# Usage:
# python main.py -m {model} -c {rgb | all}
#

import argparse
import random
import os
import time
import numpy as np

from os import PathLike

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T

from models.fast_rcnn import FasterRCNNResNet101
from data_parsing.dataloader import PrivetDataset

RAND_SEED = 7
BATCH_SIZE = 16
NUM_EPOCHS = 1


def get_model(model: str) -> torch.nn.Module:
    match model:
        case "fast_rcnn":
            return FasterRCNNResNet101()


def get_transforms(train: bool = True):
    """
    Return transforms for the data.
    """
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(p=0.5))
    return T.Compose(transforms=transforms)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data(img_dir: str | PathLike, labels_dir: str | PathLike, channels: str, batch_size: int = BATCH_SIZE) -> tuple[PrivetDataset, PrivetDataset, PrivetDataset]:
    is_multispectral = True if channels == "all" else False

    g = torch.Generator()
    g.manual_seed(RAND_SEED)

    training_data = PrivetDataset(img_dir=img_dir, labels_dir=labels_dir,
                                  is_multispectral=is_multispectral)
    validation_data = PrivetDataset(img_dir=img_dir, labels_dir=labels_dir,
                                    is_multispectral=is_multispectral)
    testing_data = PrivetDataset(img_dir=img_dir, labels_dir=labels_dir,
                                 is_multispectral=is_multispectral)

    # 80/10/10 split for data
    idxs = torch.randperm(len(training_data)).tolist()
    one_tenth = len(training_data) // 10
    eight_tenths = len(training_data) - (2 * one_tenth)
    training_data = torch.utils.data.Subset(
        training_data, idxs[:-eight_tenths])
    validation_data = torch.utils.data.Subset(
        validation_data, idxs[eight_tenths:(eight_tenths + one_tenth)])
    testing_data = torch.utils.data.Subset(testing_data, idxs[-one_tenth:])

    training_data = DataLoader(
        dataset=training_data, batch_size=batch_size, train=True, shuffle=True)
    validation_data = DataLoader(
        dataset=validation_data, train=False, shuffle=True)
    testing_data = DataLoader(dataset=testing_data, train=False, shuffle=True)
    return training_data, validation_data, testing_data


def post_results(dir: str | PathLike, num_epochs: int, batch_size: int, trained_results, test_results):
    """
    Output the results from training and testing to the specified directory.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    cts = time.localtime()
    name = f"{cts[0]}{cts[1]}{cts[2]}_{cts[3]}{cts[4]}{cts[5]}_e{num_epochs}_b{batch_size}"
    os.mkdir(os.path.join(dir, name))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="multifrequency_loader.py",
        description="Converts separate images with multiple frequencies into single tensor files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--model", help="Which model to run.\nOptions: {fast_rcnn, }")
    parser.add_argument("-c", "--channels",
                        help="Which channels to use.\nOptions: {rgb, all}")
    parser.add_argument("--img_dir", help="The outer image directory")
    parser.add_argument("--labels_dir", help="The outer label directory")
    parser.add_argument(
        "--results_dir", help="The directory to place all results")
    parser.add_argument("-e", "--epochs", type=int, nargs="+", default=[NUM_EPOCHS],
                        help="The number of epochs, space-separated (ex: python main.py -e 1 2 3)")
    parser.add_argument("-bs", "--batch_sizes", type=int, nargs="+", default=[BATCH_SIZE],
                        help="All batch sizes to use, space-separated (ex: python main.py -bs 1 2 3)")

    args = parser.parse_args()

    return args


def train(model: torch.nn.Module, train_data: DataLoader, validation_data: DataLoader, num_epochs: int):
    model.train()
    pass


def evaluate(model: torch.nn.Module, test_data: DataLoader):
    model.eval()
    pass


def main():
    print("Starting...")
    args = parse_args()

    # set up device
    device = torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.random.manual_seed(RAND_SEED)
    torch.cuda.manual_seed_all(RAND_SEED)

    for batch_size in args.batch_sizes:
        for num_epochs in args.num_epochs:
            # set up model
            model = get_model(args.model)
            model.to(device)
            # print(model)

            # set up data
            batch_size = batch_size
            train_data, validation_data, test_data = get_data(
                img_dir=args.img_dir, labels_dir=args.labels_dir, channels=args.channels, batch_size=batch_size)

            # train
            trained_results = train(model=model, train_data=train_data,
                                    validation_data=validation_data, num_epochs=num_epochs)
            print("Training complete!")

            # test
            eval_results = evaluate(model=model, test_data=test_data)

            # save results
            post_results(dir=args.results_dir,
                         trained_results=trained_results, eval_results=eval_results)


if __name__ == "__main__":
    main()
