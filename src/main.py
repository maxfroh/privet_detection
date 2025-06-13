#
# Main file
#
# Usage:
# python main.py -m {model} --img_dir {dir} --labels_dir {dir} -c {rgb | all} [-e # [...]] [-bs # [...]] [-lr # [...]]
#

import argparse
import random
import os
import time
import numpy as np

from os import PathLike
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torchvision.transforms import v2 as T
from torchvision.utils import draw_bounding_boxes

from models.fast_rcnn import FasterRCNNResNet101
from data_parsing.dataloader import PrivetDataset
from torch_references.utils import collate_fn
from torch_references.engine import train_one_epoch, evaluate

RAND_SEED = 7
BATCH_SIZE = 16
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
OPTIM_MOMENTUM = 0.9
OPTIM_WEIGHT_DECAY = 0.0005
SCHEDULER_STEP_SIZE = 30
SCHEDULER_GAMMA = 0.1

######################
#  Setup  Functions  #
######################

def get_model(model: str, num_channels: int) -> torch.nn.Module:
    match model:
        case "faster_rcnn":
            return FasterRCNNResNet101(num_channels=num_channels)


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
                                  is_multispectral=is_multispectral, transform=get_transforms())
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
        dataset=training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    validation_data = DataLoader(
        dataset=validation_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    testing_data = DataLoader(
        dataset=testing_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return training_data, validation_data, testing_data


def get_save_dir(dir: str | PathLike, num_epochs: int, batch_size: int):
    if not os.path.exists(dir):
        os.makedirs(dir)
    cts = time.localtime()
    name = f"{cts[0]}{cts[1]}{cts[2]}_{cts[3]}{cts[4]}{cts[5]}_e{num_epochs}_b{batch_size}"
    save_dir = os.path.join(dir, name)
    os.mkdir(save_dir)
    return save_dir


def save_model(model: torch.nn.Module, save_dir: str | PathLike, num_epochs: int, batch_size: int, curr_epoch: int, learning_rate: float, scheduler: LRScheduler):
    torch.save(
        {
            "epoch": curr_epoch,
            "model_state_dict": model.state_dict(),
            "scheduler": scheduler.state_dict()
        },
        os.path.join(save_dir, "models",
                     f"{batch_size}b_{curr_epoch}_of_{num_epochs}e_{learning_rate}.pt")
    )


def save_results(save_dir: str | PathLike, trained_results: dict, test_results: dict):
    """
    Output the results from training and testing to the specified directory.
    """

    with open(file=os.path.join(save_dir, "readme.txt"), mode="w", encoding="utf-8") as f:
        f.write("Test")


######################
#  Model  Functions  #
######################


def train(model: torch.nn.Module, device, train_data: DataLoader, validation_data: DataLoader, num_epochs: int, batch_size: int, lr_scheduler: LRScheduler, save_dir: str | PathLike):
    model.train()
    train_results = {}

    for e in range(num_epochs):
        start = time.time()
        print(f"Epoch {e}:")
        train_results[e] = {"train": [], "validate": []}
        for i, data in tqdm(train_data, total=len(train_data)):
            print(type(data))
            images, targets = data

            images = list(image.to(device) for image in images)
            targets = [
                {key: value.to(device) for key, value in target.items()} for target in targets]
            

            with torch.no_grad():
                loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            print(losses)
            train_results[e]["train"].append(loss_value)

            losses.backward()

            lr_scheduler.step()
            print(f"Loss: {loss_value:0.4f}")

        validate(model, device, e, validation_data, train_results)

        end = time.time()

        print(
            f"Epoch {e}/{num_epochs} trained in {((end - start) / 60):.3f} minutes")

        avg_train_loss = sum(
            train_results[e]["train"]) / len(train_results[e]["train"])
        avg_val_loss = sum(
            train_results[e]["validate"]) / len(train_results[e]["validate"])
        print(f"\tTraining Loss: {avg_train_loss:.3f}")
        print(f"\tValidation Loss: {avg_val_loss:.3f}")

        save_model(save_dir, model, e, num_epochs, batch_size, lr_scheduler)

        print("\n" + ("-" * 20) + "\n")

    return train_results


def validate(model: torch.nn.Module, device, e: int, validation_data: DataLoader, train_results: dict[int, dict[str, list[float]]]):
    for i, data in tqdm(validation_data, total=len(validation_data)):
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [
            {key: value.to(device) for key, value in target.items()} for target in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        print(losses)
        train_results[e]["validate"].append(loss_value)

        print(f"Loss: {loss_value:0.4f}")


def evaluate(model: torch.nn.Module, device, test_data: DataLoader):
    model.eval()
    test_results = {""}

    for i, data in tqdm(test_data, total=len(test_data)):
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [
            {key: value.to(device) for key, value in target.items()} for target in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        print(losses)
        test_results.append(loss_value)

        print(f"Loss: {loss_value:0.4f}")

    return test_results


def ref_train(model: torch.nn.Module, optimizer: Optimizer, train_data_loader: DataLoader, device, epoch):
    result = train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
    return result

######################
#   Main Functions   #
######################

def parse_args():
    parser = argparse.ArgumentParser(
        prog="multifrequency_loader.py",
        description="Converts separate images with multiple frequencies into single tensor files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--model", help="Which model to run.\nOptions: {faster_rcnn, }")
    parser.add_argument("-c", "--channels",
                        help="Which channels to use.\nOptions: {rgb, all}")
    parser.add_argument("--img_dir", help="The outer image directory")
    parser.add_argument("--labels_dir", help="The outer label directory")
    parser.add_argument(
        "--results_dir", help="The directory to place all results")
    parser.add_argument("-e", "--num_epochs", type=int, nargs="+", default=[NUM_EPOCHS],
                        help="The number of epochs, space-separated")
    parser.add_argument("-bs", "--batch_size", type=int, nargs="+", default=[BATCH_SIZE],
                        help="All batch sizes to use, space-separated")
    parser.add_argument("-lr", "--learning_rate", type=float, nargs="+", default=[LEARNING_RATE],
                        help="All learning rates to use, space-separated")
    parser.add_argument("--scheduler_step_size", type=float,
                        default=SCHEDULER_STEP_SIZE, help="Scheduler step size")
    parser.add_argument("--scheduler_gamma", type=float,
                        default=SCHEDULER_GAMMA, help="Scheduler gamma")
    parser.add_argument("--optimizer_momentum", type=float,
                        default=OPTIM_MOMENTUM, help="Optimizer momentum")
    parser.add_argument("--optimizer_weight_decay", type=float,
                        default=OPTIM_WEIGHT_DECAY, help="Optimizer weight decay")

    args = parser.parse_args()

    return args


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

    for batch_size in args.batch_size:
        for num_epochs in args.num_epochs:
            for learning_rate in args.learning_rate:
                # set up model
                num_channels = 3
                if args.channels == "all":
                    num_channels = 14
                model = get_model(args.model, num_channels=num_channels)
                # print(model)
                model.to(device)

                # set up data
                batch_size = batch_size
                train_data, validation_data, test_data = get_data(
                    img_dir=args.img_dir, labels_dir=args.labels_dir, channels=args.channels, batch_size=batch_size)

                # construct an optimizer
                params = [p for p in model.parameters() if p.requires_grad]
                optimizer = SGD(
                    params,
                    lr=learning_rate,
                    momentum=args.optimizer_momentum,
                    weight_decay=args.optimizer_weight_decay
                )

                # and a learning rate scheduler
                lr_scheduler = StepLR(
                    optimizer,
                    step_size=args.scheduler_step_size,
                    gamma=args.scheduler_gamma
                )

                save_dir = get_save_dir(
                    dir=args.results_dir, num_epochs=num_epochs, batch_size=batch_size)

                # train
                # trained_results = train(model=model, device=device, train_data=train_data,
                #                         validation_data=validation_data, num_epochs=num_epochs, batch_size=batch_size, lr_scheduler=lr_scheduler, save_dir=save_dir)
                
                for e in range(num_epochs):
                    trained_results = ref_train(model=model, optimizer=optimizer, train_data_loader=train_data, device=device, epoch=e)
                    print(trained_results)
                    lr_scheduler.step()
                    # evaluate on the test dataset
                    evaluate(model, validation_data, device=device)
                
                print("Training complete!")

                # test
                # eval_results = evaluate(
                #     model=model, device=device, test_data=test_data)

                # save results
                # save_results(save_dir=save_dir,
                #              trained_results=trained_results, eval_results=eval_results)


if __name__ == "__main__":
    main()
