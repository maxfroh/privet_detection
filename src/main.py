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
from typing import Union

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torchvision.transforms import v2 as T

from models.fast_rcnn import FasterRCNNResNet101
from data_parsing.dataloader import PrivetDataset, PrivetWrappedDataset
from data_parsing.graph_maker import make_graphs_and_vis
from torch_references.utils import collate_fn
from torch_references.engine import train_one_epoch, evaluate

Data = Union[Subset, PrivetDataset]

RAND_SEED = 7
BATCH_SIZE = 16
NUM_EPOCHS = 1
LEARNING_RATE = 1e-3
OPTIM_MOMENTUM = 0.9
OPTIM_WEIGHT_DECAY = 0.0005
SCHEDULER_STEP_SIZE = 30
SCHEDULER_GAMMA = 0.1

SAVE_MODELS_N = 3

######################
#  Setup  Functions  #
######################


def set_seeds(rank = 0):
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.cuda.manual_seed_all(RAND_SEED)
    torch.random.manual_seed(RAND_SEED + rank)


def get_model(model: str, num_channels: int) -> torch.nn.Module:
    if model == "faster_rcnn":
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


def get_data(img_dir: Union[str, PathLike], labels_dir: Union[str, PathLike], channels: str, batch_size: int = BATCH_SIZE, num_folds: int = 1) -> tuple[dict[int, tuple[Data, Data]], Data]:
    is_multispectral = True if channels == "all" else False

    g = torch.Generator()
    g.manual_seed(RAND_SEED)

    dataset = PrivetDataset(img_dir=img_dir, labels_dir=labels_dir,
                            is_multispectral=is_multispectral)

    fold_data: dict[int, tuple[Data, Data]] = {}
    idxs = np.random.permutation(range(len(dataset) // 5))

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    test_idx = len(dataset) - int(len(dataset) * 0.2)
    test_idxs = idxs[test_idx:]
    test_data = PrivetWrappedDataset(Subset(dataset=dataset, indices=test_idxs), test_transform)

    train_idxs = idxs[:test_idx]
    full_train_data = Subset(dataset=dataset, indices=train_idxs)

    fold = KFold(n_splits=num_folds, shuffle=True, random_state=RAND_SEED)

    splits = fold.split(full_train_data)
    for fold, (train, val) in enumerate(splits):
        training_data = PrivetWrappedDataset(Subset(dataset, train), train_transform)
        validation_data = PrivetWrappedDataset(Subset(dataset, val), test_transform)
        fold_data[fold] = (training_data, validation_data)

    return (fold_data, test_data)


def get_dataloaders(train_data: Data, val_data: Data, batch_size: int = BATCH_SIZE, rank = None, world_size = None) -> tuple[DataLoader, DataLoader]:
    if rank is not None:
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_data, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=collate_fn, pin_memory=True)
    val_dataloader = DataLoader(
        dataset=val_data, batch_size=batch_size, shuffle=False, sampler=val_sampler, collate_fn=collate_fn, pin_memory=True)
    return (train_dataloader, val_dataloader)


def get_test_dataloader(test_data: Data, batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(
        dataset=test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)


def get_save_dir(dir: Union[str, PathLike], num_epochs: int, batch_size: int, learning_rate: float, num_folds: int):
    if not os.path.exists(dir):
        os.makedirs(dir)
    cts = time.localtime()
    name = f"{cts[0]:02d}{cts[1]:02d}{cts[2]:02d}_{cts[3]:02d}{cts[4]:02d}{cts[5]:02d}_e{num_epochs}_b{batch_size}_lr{learning_rate}_kf{num_folds}"
    save_dir = os.path.join(dir, name)
    os.mkdir(save_dir)
    return save_dir


def get_model_name(num_epochs: int, batch_size: int, curr_epoch: int, learning_rate: float, fold: int):
    return f"{batch_size}b_{curr_epoch}_of_{num_epochs}e_{learning_rate}_f{fold}"


def get_model_dir(save_dir: Union[str, PathLike], model_name: str):
    return os.path.join(save_dir, "models",
                        f"{model_name}.pt")


def save_model(model: torch.nn.Module, save_dir: Union[str, PathLike], model_name: str, curr_epoch: int, optimizer: Optimizer, scheduler: LRScheduler, top_n_mAPs: list[tuple[str, float]]):
    if not os.path.exists(os.path.join(save_dir, "models")):
        os.mkdir(os.path.join(save_dir, "models"))
    torch.save(
        {
            "epoch": curr_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer,
            "scheduler": scheduler.state_dict(),
            "top_n_mAPs": top_n_mAPs,
        },
        get_model_dir(save_dir, model_name)
    )


def remove_model(save_dir: Union[str, PathLike], model_name: str):
    os.remove(get_model_dir(save_dir, model_name))


def save_results(save_dir: Union[str, PathLike], trained_results: dict[int, dict], test_results: dict, args, *, dataloaders: dict[str, DataLoader] = None, best_models: dict[list[tuple[str, float]]] = None):
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
        f.write("\n\n")
        cts = time.localtime()
        f.write(
            f"Time of writing: {cts[1]:02d}/{cts[2]:02d}/{cts[0]:02d} {cts[3]:02d}:{cts[4]:02d}:{cts[5]:02d}\n")
    torch.save(trained_results, os.path.join(save_dir, "trained_results.pt"))
    torch.save(test_results, os.path.join(save_dir, "test_results.pt"))


######################
#  Model  Functions  #
######################


def get_class_name(label: torch.Tensor):
    return {1: "privet", 2: "yew", 3: "path"}[label.item()]


def ref_train(model: torch.nn.Module, optimizer: Optimizer, train_dataloader: DataLoader, device, epoch, rank=None):
    if hasattr(train_dataloader.sampler, "set_epoch"):
        train_dataloader.sampler.set_epoch(epoch)
    result = train_one_epoch(
        model, optimizer, train_dataloader, device, epoch, print_freq=10)
    return result


def setup_fold(model_name: str, device: str, channels: str, batch_size: int, learning_rate: float, optimizer_momentum: float, optimizer_weight_decay: float, step_size: int, scheduler_gamma: float, rank = None):
    torch.cuda.set_device(rank)
    
    # set up model
    num_channels = 3 if channels == "rgb" else 14
    model = get_model(
        model_name, num_channels=num_channels)
    if rank is None:
        model.to(torch.device(device))
        # if device != "cpu":
        #     model = DataParallel(model)
    else:
        model.to(rank)
        model = DDP(model, device_ids=[rank], output_device=rank)
    # construct an optimizer
    params = [p for p in model.parameters()
              if p.requires_grad]
    optimizer = SGD(
        params,
        lr=learning_rate,
        momentum=optimizer_momentum,
        weight_decay=optimizer_weight_decay
    )
    # and a learning rate scheduler
    lr_scheduler = StepLR(
        optimizer,
        step_size=step_size,
        gamma=scheduler_gamma
    )
    print("Model created!")
    return model, optimizer, lr_scheduler

def train_with_folds(args, hyperparameters: list[Union[int, float]], fold_data: dict[int, tuple[Data, Data]], channels: str, num_folds: int, rank = None, world_size = None):
    for (batch_size, num_epochs, learning_rate, step_size, scheduler_gamma, optimizer_momentum, optimizer_weight_decay) in hyperparameters:
        trained_results = {}
        eval_results = {}
        model = None
        val_data = None
        
        # set up device
        device = "cuda" if torch.cuda.is_available() else "cpu"
           
        if rank == 0:     
            print(f"Using {device} device")
            save_dir = get_save_dir(args.results_dir, num_epochs, batch_size, learning_rate, num_folds)
        dist.barrier()

        start_time = time.time()

        for fold, (train_data, val_data) in fold_data.items():
            print(f"Fold {fold}")
            if fold not in trained_results:
                trained_results[fold] = {}
            if fold not in eval_results:
                eval_results[fold] = {}

            (train_dataloader, val_dataloader) = get_dataloaders(
                train_data, val_data, batch_size, rank=rank, world_size=world_size)

            model, optimizer, lr_scheduler = setup_fold(args.model, device, channels, batch_size, learning_rate, optimizer_momentum, optimizer_weight_decay, step_size, scheduler_gamma, rank)

            for epoch in range(num_epochs):
                
                trained_results[fold][epoch] = ref_train(
                    model=model, optimizer=optimizer, train_dataloader=train_dataloader, device=device, epoch=epoch, rank=rank)
                lr_scheduler.step()

                # evaluate on the validation dataset
                validation_result = evaluate(
                    model, val_dataloader, device=device)
                eval_results[fold][epoch] = validation_result
            
        total_time = time.time() - start_time
        if rank == 0:
            print(f"Entire run took {total_time}s")
            
            save_results(save_dir, trained_results, eval_results, args, dataloaders={"train": train_data, "validation": val_data})    
            make_graphs_and_vis(save_dir, trained_results, eval_results, val_data, model, device)


######################
#   Main Functions   #
######################

def cleanup():
    dist.destroy_process_group()


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
    parser.add_argument("--scheduler_step_size", type=float, nargs="+",
                        default=[SCHEDULER_STEP_SIZE], help="Scheduler step size")
    parser.add_argument("--scheduler_gamma", type=float, nargs="+",
                        default=[SCHEDULER_GAMMA], help="Scheduler gamma")
    parser.add_argument("--optimizer_momentum", type=float, nargs="+",
                        default=[OPTIM_MOMENTUM], help="Optimizer momentum")
    parser.add_argument("--optimizer_weight_decay", type=float, nargs="+",
                        default=[OPTIM_WEIGHT_DECAY], help="Optimizer weight decay")
    parser.add_argument("--kfold", type=int, default=1)
    parser.add_argument("--save_n_models", type=int, default=SAVE_MODELS_N,
                        help="How many best models to save per fold")

    args = parser.parse_args()

    return args


def run(rank, world_size):
    print("Starting...")
    
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method="env://")

    args = parse_args()

    set_seeds(rank)

    hyperparameters = [(batch_size, num_epochs, learning_rate, step_size, scheduler_gamma, optimizer_momentum, optimizer_weight_decay)
                       for batch_size in args.batch_size
                       for num_epochs in args.num_epochs
                       for learning_rate in args.learning_rate
                       for step_size in args.scheduler_step_size
                       for scheduler_gamma in args.scheduler_gamma
                       for optimizer_momentum in args.optimizer_momentum
                       for optimizer_weight_decay in args.optimizer_weight_decay]
    img_dir = args.img_dir
    labels_dir = args.labels_dir
    channels = args.channels
    num_folds = args.kfold

    (fold_data, test_data) = get_data(
        img_dir=img_dir, labels_dir=labels_dir, channels=channels, num_folds=num_folds)
    train_with_folds(args, hyperparameters, fold_data, channels, num_folds, rank)
    cleanup()

def main():
    world_size = 3
    mp.spawn(run,
        args=(world_size,),
        nprocs=world_size,
        join=True)


if __name__ == "__main__":
    main()
