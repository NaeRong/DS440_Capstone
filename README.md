# Tutorials Guide

Tutorial for the Low Altitude Disaster Imagery (LADI) dataset. This tutorial was originally forked from a [Penn State Learning Factory](https://www.lf.psu.edu/) capstone project.

- [Tutorials Guide](#tutorials-guide)
  - [Initial Setup](#initial-setup)
    - [Persistent System Environment Variables](#persistent-system-environment-variables)
    - [Scripts](#scripts)
  - [Tutorials - Accessing the Dataset](#tutorials---accessing-the-dataset)
    - [Getting Started](#getting-started)
    - [Clean and Validate LADI Dataset](#clean-and-validate-ladi-dataset)
  - [Tutorials - Metadata Analysis](#tutorials---metadata-analysis)
    - [iso-3166-2](#iso-3166-2)
    - [Geospatial Hurricane Analysis](#geospatial-hurricane-analysis)
    - [Tutorials - Machine Learning](#tutorials---machine-learning)
    - [PyTorch Data Loading](#pytorch-data-loading)
    - [Train and Test A Classifier](#train-and-test-a-classifier)
    - [Fine Tuning Torchvision Models](#fine-tuning-torchvision-models)
  - [Distribution Statement](#distribution-statement)

## Initial Setup

This section specifies the run order and requirements for the initial setup the repository. Other repositories in this organization may be reliant upon this setup being completed.

### Persistent System Environment Variables

Immediately after cloning this repository, [create a persistent system environment](https://superuser.com/q/284342/44051) variable titled `FLOOD_ANALYSIS_CORE` with a value of the full path to this repository root directory.

On unix there are many ways to do this, here is an example using [`/etc/profile.d`](https://unix.stackexchange.com/a/117473). Create a new file `ladi-env.sh` using `sudo vi /etc/profile.d/ladi-env.sh` and add the command to set the variable:

```bash
export FLOOD_ANALYSIS_CORE=PATH TO /ladi-tutorial
```

You can confirm `FLOOD_ANALYSIS_CORE` was set in unix by inspecting the output of `env`.

### Scripts

This is a set of boilerplate scripts describing the [normalized script pattern that GitHub uses in its projects](https://github.blog/2015-06-30-scripts-to-rule-them-all/). The [GitHub Scripts To Rule Them All](https://github.com/github/scripts-to-rule-them-all) was used as a template. Refer to the [script directory README](./script/README.md) for more details.

You will need to run these scripts in this order to download package dependencies and download all of the necessary data to get you started.

## Tutorials - Accessing the Dataset

### Getting Started

This documentation is about installing AWS tools and configuring AWS environment to download LADI dataset and load dataset in Python locally and remotely.

*Readme:* [Getting Started](./Tutorials/Get_Started.md)

### Clean and Validate LADI Dataset

This documentation is about clean the LADI dataset. For this project, we have only extracted 2000 images for training.

*Readme:* [Clean and Validate LADI Dataset](./Tutorials/Clean_Validate.md)

## Tutorials - Metadata Analysis

### iso-3166-2

This documentation performs a geospatial Analysis of the number of images taken within an administrative boundary(states) and assigns each state a color based on the number of images taken.

*Notebook:* [iso-3166-2](./Tutorials/Geospatial-Hurricane-Analysis.ipynb)

### Geospatial Hurricane Analysis

This documentation performs a geospatial Analysis of the destruction and flooding caused by Hurricanes Florence and Matthew in Florida, Georgia, and the Carolinas.

*Notebook:* [Geospatial Hurricane Analysis](./Tutorials/Geospatial-Hurricane-Analysis.ipynb)

### Tutorials - Machine Learning

### PyTorch Data Loading

[PyTorch Data Loading](./Tutorials/Pytorch_Data_Load.md)

This documentation is about loading LADI dataset in PyTorch framework including examples of writing custom `Dataset`, `Transforms` and `Dataloader`.

### Train and Test A Classifier

This documentation is about training and testing a classifier model using Convolutional Neural Network (CNN) from scratch.

*Readme:* [Train and Test A Classifier](./Tutorials/Train_Test_Classifier.md)

### Fine Tuning Torchvision Models

This documentation is about training and testing a classifier model using pre-trained ResNet and AlexNet.

*Readme:* [Fine Tuning Torchvision Models](./Tutorials/Fine_Tune_Torchvision_Models.md)

## Distribution Statement

[BSD -Clause License](https://github.com/LADI-Dataset/ladi-tutorial/blob/master/LICENSE)
