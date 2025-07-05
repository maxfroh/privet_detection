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

from models.fast_rcnn import FasterRCNNResNet101
from data_parsing.dataloader import PrivetDataset
from torch_references.utils import *
from torch_references.coco_eval import *

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


def calc_f1(P: float, R: float) -> float:
    # F1 = (2 * P * R) / (P + R)
    epsilon = 1e-10
    return (2 * P * R) / (P + R + epsilon)


######################
#                    #
######################

def visualize(save_dir: Union[str, PathLike], model: Module, device, val_data: DataLoader):
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
        all_boxes = prediction['boxes'].cpu()
        all_labels = prediction['labels'].cpu()
        all_scores = prediction['scores'].cpu()

        keep = all_scores >= score_threshold
        boxes_thr = all_boxes[keep]
        labels_thr = all_labels[keep]
        scores_thr = all_scores[keep]
        
        options = {False: (all_boxes, all_labels, all_scores), True: (boxes_thr, labels_thr, scores_thr)}
        
        for has_threshold, (boxes, labels, scores) in options.items():

            label_strings = [f"{get_class_name(label.item())}: {score.item():.4f}" for label, score in zip(
                labels, scores)]
            colors = [get_class_color(label.item())
                    for label in labels]

            boxed_img = draw_bounding_boxes((image_cpu * 255).byte().squeeze(0),
                                            boxes, labels=label_strings, 
                                            colors=colors, width=4, font_size=60,
                                            font="./resources/ARIAL.TTF", label_colors="white")

            group_imgs[has_threshold].append(boxed_img)

            if idx in [0, 1]:
                pil_img = to_pil_image(boxed_img)
                pil_img.save(os.path.join(save_dir, f"val_prediction_{idx}_{'thr' if has_threshold else 'no_thr'}.jpg"))

        for has_threshold in [False, True]:
            fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(16, 16))
            axes = axes.flatten()

            for i, axis in enumerate(axes):
                if i < len(group_imgs[has_threshold]):
                    axis.imshow(to_pil_image(group_imgs[has_threshold][i]))
                    axis.axis("off")
                else:
                    axis.remove()

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"val_boxes_{'thr' if has_threshold else 'no_thr'}.jpg"))


def plot_folds_epochs(fold_dict: dict[int, list[float]], *, title: str, y_label: str, x_label: str = "Epoch", figname: str, save_dir: Union[str, PathLike], thicken: bool = False):
    plt.figure(figsize=(10,6))
    
    avg_values = np.mean(list(fold_dict.values()), axis=0)
    std_values = np.std(list(fold_dict.values()), axis=0)
    
    plt.plot(range(len(avg_values)), avg_values, lw=4 if thicken else 2)
    plt.fill_between(range(len(avg_values)), avg_values - std_values, avg_values + std_values, alpha=0.25)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    plt.savefig(os.path.join(save_dir, figname), dpi=300, bbox_inches="tight")
   
    
def plot_folds_epochs_multiple(fold_dicts: list[dict[int, list[float]]], labels: list[str], title: str, y_label: str, figname: str, save_dir: Union[str, PathLike], thicken: list[bool] = None):
    plt.figure(figsize=(10,6))
    
    for i in range(len(fold_dicts)):
        avg_values = np.mean(list(fold_dicts[i].values()), axis=0)
        std_values = np.std(list(fold_dicts[i].values()), axis=0)
        
        thicken_line = False if thicken is None or len(thicken) == 0 else thicken[i]
        plt.plot(range(len(avg_values)), avg_values, label=f"{labels[i]}", lw=4 if thicken_line else 2)
        plt.fill_between(range(len(avg_values)), avg_values - std_values, avg_values + std_values, alpha=0.25)
        
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    plt.savefig(os.path.join(save_dir, figname), dpi=300, bbox_inches="tight")


def make_lr(save_dir: Union[str, PathLike], trained_results: dict[int, dict[int, MetricLogger]]):

    learning_rate = [np.mean([trained_results[j][i].meters["lr"].avg for j in range(len(trained_results))]) for i in range(len(trained_results[0]))]

    plt.figure(figsize=(4, 4))
    plt.title("Learning Rate per Epoch")
    plt.xlabel("Epoch Number")
    plt.ylabel("Learning Rate")

    plt.semilogy(learning_rate)

    plt.grid()

    plt.savefig(os.path.join(save_dir, "lr.png"), dpi=300, bbox_inches="tight")


def make_loss(save_dir: Union[str, PathLike], train_results: dict[int, dict[int, MetricLogger]]):

    loss_keys = {
        "loss_box_reg": "Box Training Loss",
        "loss_objectness": "Objectness Training Loss",
        "loss_classifier": "Classification Training Loss",
        "loss_rpn_box_reg": "RPN Box Training Loss"
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()

    for i, key in enumerate(loss_keys.keys()):
        axis = axes[i]
        
        epoch_vals = {}
        epochs = sorted(train_results[0].keys())
        for fold, epoch_dict in train_results.items():
            for e in epochs:
                if e not in epoch_vals:
                    epoch_vals[e] = []
                epoch_vals[e].append(epoch_dict[e].meters[key].global_avg)
        
        epoch_vals = list(epoch_vals.values())
        means = np.mean(epoch_vals, axis=1)
        std = np.std(epoch_vals, axis=1)
        
        axis.plot(epochs, means, label=f"Fold {fold}")
        axis.fill_between(epochs, means - std, means + std, 
                         alpha=0.25)
        axis.set_title(f"{key}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(f"{loss_keys[key]}")
        axis.grid(True)

    plt.suptitle("Loss Metrics per Epoch", fontsize=16)

    plt.tight_layout(rect=[0.2, 0.05, 0.98, 0.95])

    plt.savefig(os.path.join(save_dir, f"loss.png"),
                dpi=300, bbox_inches="tight")


def make_map(save_dir: Union[str, PathLike], eval_results: dict[int, dict[int, CocoEvaluator]]):
    # TxRxKxAxM
    # T = 0-10 [.5:.05:.95] iou thresholds                  => 0
    # R = 101 [0:.01:1] recall thresholds                   => ?
    # K = 3? # categories                                   => i
    # A = 4 areas for object (all, small, medium, large)    => 0
    # M = 3 [1 10 100] max detections threshold             => 2
        
    aps_05_1 = {}
    aps_05_2 = {}
    maps_05 = {}
    aps_05_095_1 = {}
    aps_05_095_2 = {}
    maps_05_095 = {}
    
    for fold, epoch_dict in eval_results.items():
        epochs = sorted(epoch_dict.keys())
        
        aps_05_1[fold] = [np.mean(epoch_dict[e].coco_eval["bbox"].eval["precision"][0, :, 0, 0, 2]) for e in epochs]
        aps_05_2[fold] = [np.mean(epoch_dict[e].coco_eval["bbox"].eval["precision"][0, :, 1, 0, 2]) for e in epochs]
        maps_05[fold] = [epoch_dict[e].coco_eval["bbox"].stats[1] for e in epochs]
        
        aps_05_095_1[fold] = [np.mean(epoch_dict[e].coco_eval["bbox"].eval["precision"][:, :, 0, 0, 2]) for e in epochs]
        aps_05_095_2[fold] = [np.mean(epoch_dict[e].coco_eval["bbox"].eval["precision"][:, :, 1, 0, 2]) for e in epochs]
        maps_05_095[fold] = [epoch_dict[e].coco_eval["bbox"].stats[0] for e in epochs]

    plot_folds_epochs_multiple(fold_dicts=[maps_05, aps_05_1, aps_05_2], 
                                labels=["mAP@0.5", "Chinese Privet AP@0.5", "Yew AP@0.5"], 
                                title="Average Precisions @ 0.5", y_label="Precision",
                                figname="ap05.png", save_dir=save_dir, thicken=[True, False, False])
    
    plot_folds_epochs_multiple(fold_dicts=[maps_05_095, aps_05_095_1, aps_05_095_2], 
                                labels=["mAP@0.5:0.95", "Chinese Privet AP@0.5:0.95", "Yew AP@0.5:0.95"], 
                                title="Average Precisions @ 0.5:0.95", y_label="Precision",
                                figname="ap05_095.png", save_dir=save_dir, thicken=[True, False, False])


def make_pr(save_dir: Union[str, PathLike], eval_results: dict[int, dict[int, CocoEvaluator]]):
    # ["precision"] = TxRxKxAxM
    # T = 0-10 [.5:.05:.95] iou thresholds                  => 0/:
    # R = 101 [0:.01:1] recall thresholds                   => ?
    # K = 3? # categories                                   => i
    # A = 4 areas for object (all, small, medium, large)    => 0
    # M = 3 [1 10 100] max detections threshold             => 2

    # ["recall"] = TxKxAxM


    # precisions = FxRxK
    # recalls = FxK
    best_epoch = np.argmax(np.mean([[eval_results[fold][e].coco_eval["bbox"].stats[0] for e in range(len(eval_results[0]))] for fold in eval_results.keys()], axis=0)).item()
    precisions = np.array([eval_results[fold][best_epoch].coco_eval["bbox"].eval["precision"][0, :, :, 0, 2] for fold in eval_results.keys()])
    # recalls = np.array([eval_results[fold][best_epoch].coco_eval["bbox"].eval["recall"][0, :, 0, 2] for fold in eval_results.keys()])
    avg_prec = np.mean(precisions, axis=-1)
    priv_prec = precisions[:, :, 0]
    yew_prec = precisions[:, :, 1]
    recThrs = eval_results[0][0].coco_eval["bbox"].params.recThrs
    # avg_rec = np.mean(recalls, axis=-1)
    # priv_rec = recalls[:, :, 0]
    # yew_rec = recalls[:, :, 1]
    
    # R
    avg_prec_mean = np.mean(avg_prec, axis=0)
    avg_prec_std = np.std(avg_prec, axis=0)
    priv_prec_mean = np.mean(priv_prec, axis=0)
    priv_prec_std = np.std(priv_prec, axis=0)
    yew_prec_mean = np.mean(yew_prec, axis=0)
    yew_prec_std = np.std(yew_prec, axis=0)

    plt.figure(figsize=(10,6))
    
    plt.plot(recThrs, avg_prec_mean, label=f"All Classes@0.5", lw=4)
    plt.fill_between(recThrs, avg_prec_mean - avg_prec_std, avg_prec_mean + avg_prec_std, alpha=0.25)
    plt.plot(recThrs, priv_prec_mean, label=f"Chinese Privet@0.5", lw=2)
    plt.fill_between(recThrs, priv_prec_mean - priv_prec_std, priv_prec_mean + priv_prec_std, alpha=0.25)
    plt.plot(recThrs, yew_prec_mean, label=f"Yew@0.5", lw=2)
    plt.fill_between(recThrs, yew_prec_mean - yew_prec_std, yew_prec_mean + yew_prec_std, alpha=0.25)

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    plt.savefig(os.path.join(save_dir, "pr.png"), dpi=300, bbox_inches="tight")


def make_f1(save_dir: Union[str, PathLike], eval_results: dict[int, dict[int, CocoEvaluator]]): #, model: Module, device: str, val_data: DataLoader):
    # TxRxKxAxM
    # T = 0-10 [.5:.05:.95] iou thresholds                  => 0/:
    # R = 101 [0:.01:1] recall thresholds                   => ?
    # K = 3? # categories                                   => i
    # A = 4 areas for object (all, small, medium, large)    => 0
    # M = 3 [1 10 100] max detections threshold             => 2
    best_epoch = np.argmax(np.mean([[eval_results[fold][e].coco_eval["bbox"].stats[0] for e in range(len(eval_results[0]))] for fold in eval_results.keys()], axis=0)).item()
    num_folds = len(eval_results)
    confidence_thrs = np.linspace(0.0, 1.0, 1001)

    f1 = np.zeros((num_folds, confidence_thrs.shape[0]))
    f1_priv = np.zeros((num_folds, confidence_thrs.shape[0]))
    f1_yew = np.zeros((num_folds, confidence_thrs.shape[0]))
    anns = [eval_results[fold][best_epoch].coco_eval["bbox"].cocoDt.anns for fold in range(num_folds)]
    coco_gt = [eval_results[fold][best_epoch].coco_eval["bbox"].cocoGt for fold in range(num_folds)]

    for fold in range(num_folds):
        for i, threshold in enumerate(confidence_thrs):
            detections = [detection for detection in anns[fold].values() if detection["score"] >= threshold]
            if len(detections) == 0:
                f1[fold][i] = 0
                f1_priv[fold][i] = 0
                f1_yew[fold][i] = 0
            else:
                new_coco_dt = coco_gt[fold].loadRes(detections)
                new_coco_eval = COCOeval(cocoGt=coco_gt[fold], cocoDt=new_coco_dt, iouType="bbox")
                new_coco_eval.evaluate()
                new_coco_eval.accumulate()

                # TxRxKxAxM => TxR => (1,)
                precision = np.mean(new_coco_eval.eval["precision"][:, :, :, 0, 2])
                privet_precision = np.mean(new_coco_eval.eval["precision"][:, :, 0, 0, 2])
                yew_precision = np.mean(new_coco_eval.eval["precision"][:, :, 1, 0, 2])
                # TxKxAxM => T => (1,)
                recall = np.mean(new_coco_eval.eval["recall"][:, :, 0, 2])
                privet_recall = np.mean(new_coco_eval.eval["recall"][:, 0, 0, 2])
                yew_recall = np.mean(new_coco_eval.eval["recall"][:, 1, 0, 2])

                f1[fold][i] = calc_f1(precision, recall)
                f1_priv[fold][i] = calc_f1(privet_precision, privet_recall)
                f1_yew[fold][i] = calc_f1(yew_precision, yew_recall)
    
    avg_f1_mean = np.mean(f1, axis=0)
    avg_f1_std = np.std(f1, axis=0)
    priv_f1_mean = np.mean(f1_priv, axis=0)
    priv_f1_std = np.std(f1_priv, axis=0)
    yew_f1_mean = np.mean(f1_yew, axis=0)
    yew_f1_std = np.std(f1_yew, axis=0)

    max_idx = np.argmax(avg_f1_mean)

    plt.figure(figsize=(10,6))
    
    plt.plot(confidence_thrs, avg_f1_mean, label=f"All Classes F1-Score | {avg_f1_mean[max_idx]:.4f} at {confidence_thrs[max_idx]}", lw=4)
    plt.fill_between(confidence_thrs, avg_f1_mean - avg_f1_std, avg_f1_mean + avg_f1_std, 
                        alpha=0.25)
    plt.plot(confidence_thrs, priv_f1_mean, label=f"Chinese Privet F1-Score", lw=2)
    plt.fill_between(confidence_thrs, priv_f1_mean - priv_f1_std, priv_f1_mean + priv_f1_std, 
                        alpha=0.25)
    plt.plot(confidence_thrs, yew_f1_mean, label=f"Yew F1-Score", lw=2)
    plt.fill_between(confidence_thrs, yew_f1_mean - yew_f1_std, yew_f1_mean + yew_f1_std, 
                        alpha=0.25)
    
    plt.title("F1-Confidence Curve")
    plt.xlabel("Confidence")
    plt.ylabel("F1-Score")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout(pad=0.4, w_pad=1.0, h_pad=1.0)
    plt.savefig(os.path.join(save_dir, "f1.png"), dpi=300, bbox_inches="tight")


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
    make_map(save_dir, eval_results)
    make_pr(save_dir, eval_results)
    make_f1(save_dir, eval_results)
    print("Done")


def make_graphs_and_vis(save_dir: Union[str, PathLike], train_results: dict, eval_results: dict, val_data: DataLoader, model: Module, device):
    save_dir = os.path.join(save_dir, "figures")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    make_graphs(save_dir, train_results, eval_results)
    make_f1(save_dir, eval_results)
    visualize(save_dir, model, device, val_data)

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

    trained_results, eval_results = load_results_data(save_dir)

    save_dir = os.path.join(save_dir, "figures")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    make_graphs(save_dir, trained_results, eval_results)

    # best_models, test_data = setup_visualize(args, save_dir, test_results)

    # visualize(save_dir, best_models, test_data)
    # states = torch.load("C:/Users/maxos/OneDrive - rit.edu/2245/reu/results/training_out_8b/models/8b_53_of_100e_0.001.pt", weights_only=False)
    # model = FasterRCNNResNet101(num_channels=3).load_state_dict(states["model_state_dict"])
    # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    
    # visualize(save_dir, model, device, train_data, val_data)


if __name__ == "__main__":
    main()
