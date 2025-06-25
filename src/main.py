#
# Main file
#
# Usage:
# python main.py -m {model} --img_dir {dir} --labels_dir {dir} -c {rgb | all} [-e # [...]] [-bs # [...]] [-lr # [...]]
#

import argparse
import random
import os
import sys
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from os import PathLike
from tqdm import tqdm
from operator import itemgetter
from sklearn.model_selection import KFold

import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torchvision.transforms import v2 as T

from models.fast_rcnn import FasterRCNNResNet101
from data_parsing.dataloader import PrivetDataset
from data_parsing.graph_maker import make_graphs_and_vis
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


def set_seeds():
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.random.manual_seed(RAND_SEED)
    torch.cuda.manual_seed_all(RAND_SEED)


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
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms=transforms)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data(img_dir: str | PathLike, labels_dir: str | PathLike, channels: str, batch_size: int = BATCH_SIZE, num_folds: int = 1) -> dict[int, tuple[DataLoader, DataLoader, DataLoader]]:
    is_multispectral = True if channels == "all" else False

    g = torch.Generator()
    g.manual_seed(RAND_SEED)

    dataset = PrivetDataset(img_dir=img_dir, labels_dir=labels_dir,
                            is_multispectral=is_multispectral)

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)
        
    dls: dict[int, tuple[DataLoader, DataLoader, DataLoader]] = {}
    idxs = np.random.permutation(range(len(dataset)))

    if num_folds > 1:
        # 20%
        test_idx = len(dataset) - int(len(dataset) * 0.1)
        train_idxs = idxs[:test_idx]
        test_idxs = idxs[test_idx:]
        full_train_data = Subset(dataset=dataset, indices=train_idxs)
        test_data = Subset(dataset=dataset, indices=test_idxs)
        test_data = DataLoader(
            dataset=test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        fold = KFold(n_splits=num_folds, shuffle=True, random_state=RAND_SEED)

        splits = fold.split(full_train_data)
        for fold, (train, val) in enumerate(splits):
            training_data = Subset(dataset, train)
            validation_data = Subset(dataset, val)
            training_data.transform = train_transform
            validation_data.transform = test_transform
            training_data = DataLoader(
                dataset=training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            validation_data = DataLoader(
                dataset=validation_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            dls[fold] = (training_data, validation_data, test_data)

    else:
        # 80/10/10 split for data
        idxs = torch.randperm(len(dataset)).tolist()
        one_tenth = len(dataset) // 10
        training_data = Subset(
            dataset, idxs[:one_tenth * 8])
        training_data.dataset.transform = train_transform
        validation_data = Subset(
            dataset, idxs[one_tenth * 8:one_tenth * 9])
        validation_data.dataset.transform = test_transform
        test_data = Subset(dataset, idxs[one_tenth * 9:])
        test_data.dataset.transform = test_transform

        training_data = DataLoader(
            dataset=training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        validation_data = DataLoader(
            dataset=validation_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_data = DataLoader(
            dataset=test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        dls[0] = (training_data, validation_data, test_data)

    return dls


def get_save_dir(dir: str | PathLike, num_epochs: int, batch_size: int, learning_rate: float):
    if not os.path.exists(dir):
        os.makedirs(dir)
    cts = time.localtime()
    name = f"{cts[0]:02d}{cts[1]:02d}{cts[2]:02d}_{cts[3]:02d}{cts[4]:02d}{cts[5]:02d}_e{num_epochs}_b{batch_size}_lr{learning_rate}"
    save_dir = os.path.join(dir, name)
    os.mkdir(save_dir)
    return save_dir


def get_model_name(num_epochs: int, batch_size: int, curr_epoch: int, learning_rate: float, fold: int):
    return f"{batch_size}b_{curr_epoch}_of_{num_epochs}e_{learning_rate}_f{fold}"


def get_model_dir(save_dir: str | PathLike, model_name: str):
    return os.path.join(save_dir, "models",
                        f"{model_name}.pt")


def save_model(model: torch.nn.Module, save_dir: str | PathLike, model_name: str, curr_epoch: int, optimizer: Optimizer, scheduler: LRScheduler, top_5_mAPs: list[tuple[str, float]]):
    if not os.path.exists(os.path.join(save_dir, "models")):
        os.mkdir(os.path.join(save_dir, "models"))
    torch.save(
        {
            "epoch": curr_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer,
            "scheduler": scheduler.state_dict(),
            "top_5_mAPs": top_5_mAPs,
        },
        get_model_dir(save_dir, model_name)
    )


def remove_model(save_dir: str | PathLike, model_name: str):
    os.remove(get_model_dir(save_dir, model_name))


def save_results(save_dir: str | PathLike, trained_results: dict[int, dict], test_results: dict, args, *, dataloaders: dict[str, DataLoader] = None, best_models: list[tuple[str, float]] = None):
    """
    Output the results from training and testing to the specified directory.
    """
    with open(file=os.path.join(save_dir, "readme.txt"), mode="w", encoding="utf-8") as f:
        f.write("This model has been trained with the following parameters:\n")
        for arg in vars(args):
            line = f"\t{arg}: {getattr(args, arg)}\n"
            f.write(line)
        if dataloaders:
            for name, dataloader in dataloaders.items():
                f.write(f"{name} size: {len(dataloader)}\n")
        if best_models:
            f.write("Best models:")
            for item in best_models:
                f.write(f"\t- Model: {item[0]} | mAP@0.5: {item[1]}")
    torch.save(trained_results, os.path.join(save_dir, "trained_results.pt"))
    torch.save(test_results, os.path.join(save_dir, "test_results.pt"))

######################
#  Model  Functions  #
######################


def get_class_name(label: torch.Tensor):
    return {1: "privet", 2: "yew", 3: "path"}[label.item()]


def ref_train(model: torch.nn.Module, optimizer: Optimizer, train_data_loader: DataLoader, device, epoch):
    result = train_one_epoch(
        model, optimizer, train_data_loader, device, epoch, print_freq=10)
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
    parser.add_argument("--kfold", type=int, default=1)

    args = parser.parse_args()

    return args


def main():
    print("Starting...")
    args = parse_args()

    # set up device
    device = torch.accelerator.current_accelerator().type \
        if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    set_seeds()

    num_folds = args.kfold

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
                dataloaders = get_data(
                    img_dir=args.img_dir, labels_dir=args.labels_dir, channels=args.channels, batch_size=batch_size, num_folds=num_folds)
                for fold, (train_data, validation_data, test_data) in enumerate(dataloaders.values()):
                    print(f"Starting Fold {fold}")

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
                        dir=args.results_dir, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)

                    trained_results = {}
                    eval_results = {}
                    top_5_mAPs: list[tuple[str, float]] = []

                    start_time = time.time()

                    for epoch in range(num_epochs):
                        print(f"Epoch {epoch}/{num_epochs}")
                        trained_results[epoch] = ref_train(
                            model=model, optimizer=optimizer, train_data_loader=train_data, device=device, epoch=epoch)
                        lr_scheduler.step()

                        # evaluate on the validation dataset
                        validation_result = evaluate(
                            model, validation_data, device=device)
                        eval_results[epoch] = validation_result

                        mAP = validation_result.coco_eval['bbox'].stats[1]

                        # save results
                        save_results(save_dir=save_dir,
                                     trained_results=trained_results, test_results=eval_results, args=args)

                        # save model if in top 5
                        model_name = get_model_name(num_epochs=num_epochs, batch_size=batch_size, curr_epoch=epoch, learning_rate=learning_rate, fold=fold)
                        if len(top_5_mAPs) < 5:
                            save_model(model=model, save_dir=save_dir, model_name=model_name, curr_epoch=epoch,
                                       optimizer=optimizer, scheduler=lr_scheduler, top_5_mAPs=top_5_mAPs)
                            top_5_mAPs.append((model_name, mAP))
                            top_5_mAPs.sort(key=itemgetter(1), reverse=True)
                        elif mAP > top_5_mAPs[-1][1]:
                            for i in range(len(top_5_mAPs)):
                                if mAP > top_5_mAPs[i][1]:
                                    top_5_mAPs.insert(i, (model_name, mAP))
                                    break
                            name_to_delete = top_5_mAPs[-1][0]
                            try:
                                os.remove(get_model_dir(
                                    save_dir, name_to_delete))
                            except:
                                continue
                            save_model(model=model, save_dir=save_dir, model_name=model_name, curr_epoch=epoch,
                                       optimizer=optimizer, scheduler=lr_scheduler, top_5_mAPs=top_5_mAPs)
                            top_5_mAPs = top_5_mAPs[:-1]

                    print("\n\nTraining complete!\n\n")

                    # test
                    eval_results[fold][-1] = evaluate(model,
                                                test_data, device=device)
                    save_results(save_dir=save_dir,
                                 trained_results=trained_results, test_results=eval_results, args=args, dataloaders={"training_data": train_data, "validation_data": validation_data, "testing_data": test_data}, best_models=top_5_mAPs)

                    total_time = time.time() - start_time
                    print(f"Entire run took {total_time}s")

                    make_graphs_and_vis(save_dir=save_dir,
                                        trained_results=trained_results, test_results=eval_results, best_models=top_5_mAPs, test_data=test_data)

                print("Complete!")


if __name__ == "__main__":
    main()
