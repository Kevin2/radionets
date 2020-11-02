# radionets [![Build Status](https://travis-ci.com/Kevin2/radionets.svg?branch=master)](https://travis-ci.org/kevin2/radionets)

### Imaging radio interferometric data with neural networks

Executables to simulate and analyze radio interferometric data in python. The goal is to reconstruct (image) calibrated observations with convolutional neural networks. 
This repository is build up as a python package. After cloning you can install it with
`pip install .` after navigating to the folder.
While installing you may experience some problems with cartopy. In this case you have to install a proj and a geos library before:
```
$ conda env create -f environment.yml
```

At the moment the repository covers the following blocks:

* `radionets_simulations <...>`
  This script is used to simulate radio interferometric datasets for the training of deep learning models.
* `radionets_training <...>`
  This script is used to train a model on events with known truth
  values for the target variable, usually monte carlo simulations.
* `radionets_evaluation <...>`
  This script is used to evaluate the performance of the trained deep learning models.
* `radionets_reconstruction <...>`
  This script is used to reconstruct radio interferometric data using a trained deep learning model.

Default configuration files can be found in the examples directory.

## simulations

Functions to simulate and illustrate radio interferometric observations.

* Define antenna arrays
* Calculate baselines
* Simulate (uv)-coverages
* Create (uv)-masks
* Illustrate uv-coverages and baselines for different observations

## mnist_cnn

Feasibility study to test analysis strategies with convolutional neural networks.

* Reconstruct handwritten digits from their sampled Fourier spectrum
* Simulated VLBA observations used for sampling
* Simple CNN model for reconstruction and retransformation

All analysis steps can be run using the Makefile inside the mnist_cnn directory.
The different steps for an example analysis are:
1. mnist_fft: rescale and create the Fourier transformation of the mnist images
2. mnist_samp: sample the Fourier space with simulated (uv)-coverages
3. calc_normalization: calculate normalization factors to normalize train and valid dataset
4. cnn_training: train the convolutional neural network, many options are available here

## pointsources

Simulation of pointsource delta peaks. Reconstruction with UNet architectures. Different functions to 
evaluate the reconstruction results.

## gauss

Simulation of pointlike gauss sources. Reconstruction with UNet architectures. Different functions to 
evaluate the reconstruction results.

## Versions used and tested

* Python >= 3.6
* pyTorch >= 1.2.0
* torchvision >= 0.4.0
* cudatoolkit >= 9.2
