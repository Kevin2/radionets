# radionets [![Build Status](https://travis-ci.com/Kevin2/radionets.svg?branch=master)](https://travis-ci.org/kevin2/radionets)

## Imaging Radio Interferometric Data with Neural Networks

Executables to simulate and analyze radio interferometric data in python. The goal is to reconstruct calibrated observations with convolutional neural networks to create high resolution images. 

Analysis strategies leading to a reproducible processing and evaluation of data recorded by radio interferometers:
* Simulation of datasets
* Training of deep learning models
* Reconstruction of radio interferometric data

## Installation

This repository is build up as a python package. It is recommended to create an own conda environment to handle the dependencies of the packages. You can create one by running the following command in this repository:
```
$ conda env create -f environment.yaml
```
Depending on your `cuda` version you have to specify the `cudatoolkit` version used by `pytorch`. If you working on machines with `cuda` versions < 10.2, please
change the version number in the environment.yml file.

## Usage

For each tasks executables are intstalled to your `PATH`. Each of them take `toml` configuration files as input, to manage data paths and options.
Simulated data is saved in `hdf5`, trained models are saved as `pickle` files.

* `radionets_simulate <...>`
  This script is used to simulate radio interferometric datasets for the training of deep learning models.
* `radionets_training <...>`
  This script is used to train a model on events with known truth
  values for the target variable, usually monte carlo simulations.
* `radionets_evaluation <...>`
  This script is used to evaluate the performance of the trained deep learning models.
* `radionets_reconstruction <...>`
  This script is used to reconstruct radio interferometric data using a trained deep learning model.


## Structure of the Repository

### dl_framework

The used deep learning framework is based on [pytorch](https://pytorch.org/) and [fastai](https://www.fast.ai/).
An introduction to neural networks and an overview of the use of fastai to train deep learning models can be found in [Practical Deep Learning for Coders, v3](https://course.fast.ai/index.html) and [fastbook](https://github.com/fastai/fastbook).

### dl_training

Functions for handling the different training options.

### simulations

Functions to simulate and illustrate radio interferometric observations. At the moment simulations based on the MNIST dataset and 
simulations of Gaussian sources are possible.

### evaluation

'''
evaluate training process with different approaches

ms-ssim
blob-detection
'''

## Contributors

* Kevin Schmidt [@Kevin2](https://github.com/Kevin2)
* Felix Geyer [@FritzGeise](https://github.com/FritzGeise)
* Kevin Laudamus [@K-Lauda](https://github.com/K-Lauda)
* Emiliano Miranda [@emilianozm24](https://github.com/emilianozm24)
* Maximilian Büchel [@MaxBue](https://github.com/MaxBue)

## Versions used and tested

* Python >= 3.6
* pyTorch >= 1.2.0
* torchvision >= 0.4.0
* cudatoolkit >= 9.2