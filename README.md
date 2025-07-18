# privet_detection

This is the project containing the Faster R-CNN used for the UNT AI Summer Research Program project "Identification of *Ligustrum sinense* (Chinese Privet) using deep learning models and UAV imagery" under John South.

## What is Faster R-CNN?
Read more about R-CNNs here: ["Rich feature hierarchies for accurate object detection and semantic segmentation" (Girshick et al., 2014)](https://arxiv.org/pdf/1311.2524).

Read more about Faster R-CNN here: ["Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks" (Ren et al., 2016)](https://arxiv.org/pdf/1506.01497).

## Project Structure
>### Contents
>- [Data](#data)
>- [Data Parsing](#srcdata_parsing)
>   - [Multifrequency Conversion](#multifrequency_loaderpy)
>   - [Graph Maker](#graph_makerpy)
>   - [PrivetDataset](#dataloaderpy)
>- [Models](#srcmodels)
>- [TorchReferences](#srctorch_references)
>- [`main.py`](#srcmainpy)
>- [`run.slurm`](#srcrunslurm)


### data
The data folder contains a selection of some images used in the training process (under `/images`) and all labels used for training (under `/labels`).

### src/data_parsing
These files a related to data handling, whether that be for preprocessing, training, or post-training evaluation.

#### multifrequency_loader.py
This file takes in one or more image directories as parameters of the `-dirs-in` argument. It expects the image directories to contain (almost) identically named files in JPG or TIF format. These files are the RGB, RE, and IR images. These should be the standard output of the DJI Mavic 3M.

#### graph_maker.py
The graph maker file contains all functions to create graphs and visuals once the model has completed training.

#### dataloader.py
This file contains the `PrivetDataset` class, which loads all images and labels and makes them into usable objects for the Faster R-CNN model. 

It also contains the `PrivetWrappedDataset` class, which is used to add a transform to the data after the `PrivetDataset` object has already been initialized. This is necessary because the dataset is made without any transforms, so that it can be permuted into different configurations for k-fold validation.

### src/models
The `fast_rcnn.py` file contains the code to create a Faster R-CNN with ResNet101 backbone. 

Read more about this here: 
- [Faster R-CNN (torchvision docs)](https://docs.pytorch.org/vision/main/models/faster_rcnn.html)
- [Faster R-CNN Example (torchvision tutorial)](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
- [ResNet101 (torchvision docs)](https://docs.pytorch.org/vision/2.0/models/generated/torchvision.models.resnet101.html)

### src/torch_references
This folder contains torchvision-created files useful for training and evaluating Faster R-CNNs. They are taken from [vision.references.detection.*](https://github.com/pytorch/vision/tree/main/references/detection). The original authors do better than I ever could.

### src/main.py
This file contains all code to create, train, and evaluate one or more Faster R-CNN models via the command line. 

At the top of the file, you will see functions to load data, set up the model and related objects, and handle other configuration things. 

At the bottom, the main code to start the program and parse arguments.

In the middle, the training loops. There are two possible paths, one with k-fold cross validation and one with a simple train/test split. They function identically, creating a model with the given hyperparameters, training and evaluating in a loop using `torch_references` functions, and ending by creating graphs via `graph_maker.py`. 

### src/run.slurm
This file is used to run the code via the [Texas Advanced Computing Center (TACC)](https://tacc.utexas.edu/). Please consult [TACC documentation](https://docs.tacc.utexas.edu/hpc/lonestar6/).
____

Questions? Contact Max Frohman at mbf1102@rit.edu.