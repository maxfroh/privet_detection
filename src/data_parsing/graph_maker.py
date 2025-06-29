from models.fast_rcnn import FasterRCNNResNet101
from data_parsing.dataloader import PrivetDataset
from torch_references.utils import *
from torch_references.coco_eval import *
import os
import sys
import re
import random
import argparse
import pycocotools
import matplotlib.pyplot as plt
import numpy as np

from os import PathLike
from pathlib import Path
from collections import defaultdict, deque
from operator import itemgetter
from typing import Union

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.serialization import safe_globals
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import to_pil_image
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

sys.path.append(str(Path(__file__).parent.parent))


RAND_SEED = 7

######################
#                    #
######################


def get_eval_vals(test_results: dict):
    """
    <ol start="0">
    <li>AP@IoU=0.50:0.95 (all)</li>
    <li>AP@IoU=0.50 (all)</li>
    <li>AP@IoU=0.75 (all)</li>
    <li>AP@IoU=0.50:0.95 (small)</li>
    <li>AP@IoU=0.50:0.95 (medium)</li>
    <li>AP@IoU=0.50:0.95 (large)</li>
    <li>AR@IoU=0.50:0.95 (all)</li>
    <li>AR@IoU=0.50 (all)</li>
    <li>AR@IoU=0.75 (all)</li>
    <li>AR@IoU=0.50:0.95 (small)</li>
    <li>AR@IoU=0.50:0.95 (medium)</li>
    <li>AR@IoU=0.50:0.95 (large)</li>
    </ol>
    """
    arr = np.ndarray((12, len(test_results)))
    for i in range(len(test_results) - 1):
        result = test_results[i].coco_eval["bbox"].stats
        arr[:, i] = result
    result = test_results[-1].coco_eval["bbox"].stats
    arr[:, -1] = result
    return arr


def get_class_name(i):
    return {0: "bg", 1: "privet", 2: "yew"}[i]


def get_class_color(i):
    return {0: "black", 1: "red", 2: "salmon"}[i]


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


def get_model(model: str, num_channels: int) -> torch.nn.Module:
    if model == "faster_rcnn":
        return FasterRCNNResNet101(num_channels=num_channels)


def get_model_dir(save_dir: Union[str, PathLike], model_name: str):
    return os.path.join(save_dir, "models",
                        f"{model_name}.pt")


def get_best_model(save_dir, best_models):
    model = get_model("faster_rcnn", 3)
    best_model_name = sorted(
        best_models, key=itemgetter(1), reverse=True)[0][0]

    state_dict = torch.load(get_model_dir(
        save_dir, best_model_name), weights_only=False)["model_state_dict"]
    model.load_state_dict(state_dict)
    return model


######################
#                    #
######################

def visualize(save_dir: Union[str, PathLike], model: Module, device, train_data: DataLoader, val_data: DataLoader):
    model.to(device)
    model.eval()

    num_images = 16
    grid_rows, grid_cols = 4, 4
    score_threshold = 0.5
    group_imgs = {False: [], True: []}

    for idx, (image, _) in enumerate(val_data):
        if idx >= num_images:
            break

        image = image.to(device)
        with torch.no_grad():
            prediction = model([image])[0]

        image_cpu = image.cpu()
        boxes = prediction['boxes'].cpu()
        labels = prediction['labels'].cpu()
        scores = prediction['scores'].cpu()

        keep = scores >= score_threshold
        boxes_thr = boxes[keep]
        labels_thr = labels[keep]
        scores_thr = scores[keep]
        
        options = {False: (boxes, labels, scores), True: (boxes_thr, labels_thr, scores_thr)}
        
        for has_threshold, (boxes, labels, scores) in options.items():

            label_strings = [f"{get_class_name(label.item())}: {score.item():.4f}" for label, score in zip(
                prediction["labels"], prediction["scores"])]
            colors = [get_class_color(label.item())
                    for label in prediction["labels"]]

            boxed_img = draw_bounding_boxes((image_cpu * 255).byte().squeeze(0),
                                            boxes, labels=label_strings, 
                                            colors=colors, width=4, font_size=16,
                                            font="arial.ttf")

            group_imgs[has_threshold].append(boxed_img)

            if idx in [0, 1]:
                pil_img = to_pil_image(boxed_img)
                pil_img.save(os.path.join(save_dir, f"val_prediction_{idx}_{"thr" if has_threshold else "no_thr"}.jpg"))

        for has_threshold in [False, True]:
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 16))
            axes = axes.flatten()

            for i, axis in enumerate(axes):
                if i < len(group_imgs[has_threshold]):
                    axis.imshow(to_pil_image(group_imgs[has_threshold][i]))
                    axis.axis('off')
                else:
                    axis.remove()

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"val_boxes_{"thr" if has_threshold else "no_thr"}.jpg"))


def make_lr(save_dir: Union[str, PathLike], trained_results: dict[int, dict[int, MetricLogger]]):
    learning_rate = np.array([[i, trained_results[0][i].meters["lr"].avg]
                             for i in range(len(trained_results))])

    plt.figure(figsize=(4, 4))
    plt.title("Learning Rate per Epoch")
    plt.xlabel("Epoch Number")
    plt.ylabel("Learning Rate")

    plt.semilogy(learning_rate[:, 0], learning_rate[:, 1])

    plt.grid()

    plt.savefig(os.path.join(save_dir, "lr.png"), dpi=300, bbox_inches="tight")


def make_loss(save_dir: Union[str, PathLike], train_results: dict[int, dict[int, MetricLogger]]):

    loss_keys = {
        "loss_box_reg": "Box Training Loss",
        "loss_objectness": "Objectness Training Loss",
        "loss_classifier": "Classification Training Loss",
        "loss_rpn_box_reg": "RPN Box Training Loss"
    }

    cmap = plt.cm.get_cmap("cividis", len(train_results))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, key in enumerate(loss_keys.keys()):
        axis = axes[i]
        for fold, epoch_dict in train_results.items():
            epochs = sorted(epoch_dict.keys())
            values = [epoch_dict[e].meters[key].global_avg for e in epochs]
            axis.plot(epochs, values, label=f"Fold {fold}")
            axis.set_title(f"{key}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(f"{loss_keys[key]}")
            axis.grid(True)
            axis.legend()

    plt.suptitle("Loss Metrics per Epoch", fontsize=16)

    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)

    plt.savefig(os.path.join(save_dir, f"loss.png"),
                dpi=300, bbox_inches="tight")


def make_pr(save_dir: Union[str, PathLike], eval_results: dict[int, dict[int, CocoEvaluator]]):
    metrics = {"recall": "Recall", "precision": "Precision",
               "mAP_0.5": "mAP@0.5", "mAP_0.5_0.95": "mAP@0.5:0.95"}
    cmap = plt.cm.get_cmap("viridis", len(eval_results))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        axis = axes[i]

        fold_values = {}
        for fold, epoch_dict in eval_results.items():
            epochs = sorted(epoch_dict.keys())
            values = [epoch_dict[e].coco_eval["bbox"].stats[i]
                      for e in epochs]  # Adjust index based on metric
            fold_values[fold] = values

        avg_values = np.mean(list(fold_values.values()), axis=0)
        std_values = np.std(list(fold_values.values()), axis=0)
        min_values = np.min(list(fold_values.values()), axis=0)
        max_values = np.max(list(fold_values.values()), axis=0)

        axis.plot(epochs, avg_values, label="Average",
                  color="black", linestyle="-", lw=2)
        # std
        axis.fill_between(epochs, avg_values - std_values, avg_values +
                          std_values, color="gray", alpha=0.5, label="±1 Std")
        # min/max
        axis.fill_between(epochs, min_values, max_values,
                          color="lightgray", alpha=0.3, label="Min/Maxis")

        axis.set_title(f"{metrics[metric]} per Epoch")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metrics[metric])
        axis.grid(True)
        axis.legend()

    plt.suptitle("Evaluation Metrics per Epoch (Across Folds)", fontsize=16)

    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)

    plt.savefig(os.path.join(save_dir, "pr.png"),
                dpi=300, bbox_inches="tight")


def make_f1(save_dir: Union[str, PathLike], eval_results: dict[int, dict[int, CocoEvaluator]]):
    metrics = ['f1_overall', 'f1_per_class']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # So we can index them with a single loop

    for i, metric in enumerate(metrics):
        axis = axes[i]

        # Collect F1 scores for each fold and epoch
        fold_values = {}

        for fold, epoch_dict in eval_results.items():
            epochs = sorted(epoch_dict.keys())
            values = []

            for e in epochs:
                evaluator = epoch_dict[e]
                coco_eval = evaluator.coco_eval['bbox']

                if metric == 'f1_overall':
                    precision = coco_eval.stats[1]
                    recall = coco_eval.stats[0]
                    f1 = 2 * (precision * recall) / (precision + recall)
                    values.append(f1)

                elif metric == 'f1_per_class':
                    f1_class = coco_eval.stats[3:]
                    values.append(f1_class)

            fold_values[fold] = values

        avg_values = np.mean(list(fold_values.values()), axis=0)
        std_values = np.std(list(fold_values.values()), axis=0)

        if metric == 'f1_overall':
            axis.plot(epochs, avg_values, label='Average F1',
                      color='black', linestyle='-', lw=2)
            axis.fill_between(epochs, avg_values - std_values, avg_values +
                              std_values, color='gray', alpha=0.5, label='±1 Std')
            axis.set_title('Overall F1 Confidence Curve (Across Folds)')
            axis.set_ylabel('F1 Score')

        elif metric == 'f1_per_class':
            for class_idx in range(len(avg_values[0])):
                class_avg = np.array([v[class_idx] for v in avg_values])
                class_std = np.array([v[class_idx] for v in std_values])
                axis.plot(epochs, class_avg,
                          label=f'Class {class_idx} F1', lw=2)
                axis.fill_between(epochs, class_avg - class_std,
                                  class_avg + class_std, alpha=0.3)

            axis.set_title('Per-Class F1 Confidence Curve (Across Folds)')
            axis.set_ylabel('F1 Score')

        axis.set_xlabel('Epoch')
        axis.grid(True)
        axis.legend()

    plt.suptitle('F1 Confidence Curves (Across Folds)', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(os.path.join(save_dir, "f1.png"),
                dpi=300, bbox_inches="tight")


######################
#                    #
######################


def load_results_data(save_dir: Union[str, PathLike]):
    with safe_globals([MetricLogger, SmoothedValue, defaultdict, deque, CocoEvaluator, COCOeval, pycocotools.coco.COCO, list, np._core.multiarray.scalar, np.dtype, np.dtypes.Int64DType]):
        trained_results = torch.load(os.path.join(
            save_dir, "trained_results.pt"), weights_only=False)
        test_results = torch.load(os.path.join(
            save_dir, "test_results.pt"), weights_only=False)
    return trained_results, test_results


def make_graphs(save_dir: Union[str, PathLike], trained_results: dict, eval_results: dict):
    make_lr(save_dir, trained_results)
    make_loss(save_dir, trained_results)
    make_pr(save_dir, eval_results)
    make_f1(save_dir, eval_results)
    print("Done")


def make_graphs_and_vis(save_dir: Union[str, PathLike], train_results: dict, eval_results: dict, train_data: DataLoader, val_data: DataLoader, model: Module, device):
    save_dir = os.path.join(save_dir, "figures")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    make_graphs(save_dir, train_results, eval_results)
    visualize(save_dir, model, device, train_data, val_data)

######################
#                    #
######################


def setup_visualize(args, save_dir, test_results):
    batch_size = int(re.findall(r"(?<=b)\d{1,2}(?=_lr)", save_dir)[0])
    best_models = [f.replace(".pt", "")
                   for f in os.listdir(os.path.join(save_dir, "models"))]
    indices = [(i, int(re.findall(r"\d{1,2}(?=_of_)", best_models[i])[
                0]) - 1) for i in range(len(best_models))]
    top_5_mAP = []
    for best_models_idx, actual_idx in indices:
        mAP = test_results[actual_idx].coco_eval["bbox"].stats[1]
        top_5_mAP.append((best_models[best_models_idx], mAP))

    is_multispectral = True if args.channels == "all" else False
    test_data = PrivetDataset(img_dir=args.img_dir, labels_dir=args.labels_dir,
                              is_multispectral=is_multispectral)

    idxs = torch.randperm(len(test_data)).tolist()
    one_tenth = len(test_data) // 10
    eight_tenths = len(test_data) - (2 * one_tenth)
    test_data = torch.utils.data.Subset(
        test_data, idxs[(eight_tenths + one_tenth):])

    test_data = DataLoader(
        dataset=test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    return top_5_mAP, test_data


def parse_args():
    parser = argparse.ArgumentParser(
        prog="multifrequency_loader.py",
        description="Converts separate images with multiple frequencies into single tensor files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--model", help="Which model to run.\nOptions: {faster_rcnn, }", default="faster_rcnn")
    parser.add_argument("-c", "--channels",
                        help="Which channels to use.\nOptions: {rgb, all}", default="rgb")
    parser.add_argument("--img_dir", help="The outer image directory")
    parser.add_argument("--labels_dir", help="The outer label directory")
    parser.add_argument(
        "--save_dir", help="The directory results were saved in")

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    torch.random.manual_seed(RAND_SEED)
    torch.cuda.manual_seed_all(RAND_SEED)

    save_dir = args.save_dir

    trained_results, test_results = load_results_data(save_dir)

    save_dir = os.path.join(save_dir, "figures")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # make_graphs(save_dir, trained_results, test_results)

    # best_models, test_data = setup_visualize(args, save_dir, test_results)

    # visualize(save_dir, best_models, test_data)


if __name__ == "__main__":
    main()
