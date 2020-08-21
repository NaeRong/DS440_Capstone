# Tutorials Guide

Tutorial for the Low Altitude Disaster Imagery (LADI) dataset. This tutorial was originally forked from a [Penn State Learning Factory](https://www.lf.psu.edu/) capstone project

- [Tutorials Guide](#tutorials-guide)
  - [Getting Started](#getting-started)
  - [Clean and Validate LADI Dataset](#clean-and-validate-ladi-dataset)
  - [iso-3166-2](#iso-3166-2)
  - [Geospatial Hurricane Analysis](#Geospatial-Hurricane-Analysis)
  - [PyTorch Data Loading](#pytorch-data-loading)
  - [Train and Test A Classifier](#train-and-test-a-classifier)
  - [Fine Tuning Torchvision Models](#fine-tuning-torchvision-models)
  - [Scripts](#Scripts)
  - [Distribution Statement](#distribution-statement)

## Getting Started

[Getting Started](Get_Started.md)

This documentation is about installing AWS tools and configuring AWS environment to download LADI dataset and load dataset in Python locally and remotely.

## Clean and Validate LADI Dataset

[Clean and Validate LADI Dataset](Clean_Validate.md)

This documentation is about clean the LADI dataset. For this project, we have only extracted 2000 images for training.

## iso-3166-2

[iso-3166-2](iso-3166-2.ipynb)

This documentation performs a geospatial Analysis of the number of images taken within an administrative boundary(states) and assigns each state a color based on the number of images taken.

## Geospatial Hurricane Analysis

[Geospatial Hurricane Analysis](Geospatial-Hurricane-Analysis.ipynb)

This documentation performs a geospatial Analysis of the destruction and flooding caused by Hurricanes Florence and Matthew in Florida, Georgia, and the Carolinas.

## PyTorch Data Loading

[PyTorch Data Loading](Pytorch_Data_Load.md)

This documentation is about loading LADI dataset in PyTorch framework including examples of writing custom `Dataset`, `Transforms` and `Dataloader`.

## Train and Test A Classifier

[Train and Test A Classifier](Train_Test_Classifier.md)

This documentation is about training and testing a classifier model using Convolutional Neural Network (CNN) from scratch.

## Fine Tuning Torchvision Models

[Fine Tuning Torchvision Models](Fine_Tune_Torchvision_Models.md)

This documentation is about training and testing a classifier model using pre-trained ResNet and AlexNet.

## Scripts

[Scripts](../scripts/README.md)

This is a set of boilerplate scripts describing the normalized script pattern that GitHub uses in its projects. The GitHub Scripts To Rule Them All was used as a template. These scripts will download all of the necessary data to get you started.

## Distribution Statement

[BSD -Clause License](https://github.com/LADI-Dataset/ladi-tutorial/blob/master/LICENSE)
