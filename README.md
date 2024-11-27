# Apparel-Image-Classification-with-WideResNet
This project implements apparel image classification using the WideResNet architecture on the Fashion MNIST dataset. The implementation utilizes PyTorch and includes data preprocessing, model training, evaluation, and visualization.

## Table of Contents
- [Introduction](#introduction)
- [Project Pipeline](#project-pipeline)
- [Data](#data)
- [Model](#model)
- [Results](#results)
- [How to Run](#how-to-run)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)

## Introduction
The goal of this project is to classify images of clothing items into 10 categories using the Fashion MNIST dataset. The project explores the WideResNet model, a variation of ResNet with increased width, to improve classification accuracy.

## Project Pipeline
1. Load and preprocess the Fashion MNIST dataset.
2. Build the WideResNet architecture.
3. Train the model on the training dataset.
4. Evaluate the model's performance on the test dataset.
5. Visualize training metrics and predictions.

## Data
The dataset used is the Fashion MNIST dataset, which contains 70,000 grayscale images of 28x28 pixels. It is split into:
- **Training set**: 60,000 images
- **Test set**: 10,000 images

## Model
The WideResNet model architecture includes:
- Residual and convolutional blocks
- Batch normalization
- Dropout for regularization
- Global average pooling
- Fully connected layers

### Training Configuration
- Batch size: `32`
- Epochs: `40`
- Learning rate: `0.01`
- Early stopping with a patience of `2`

## Results
- **Validation Accuracy**: 90.5%
- **Test Loss**: 0.258
- **Test Accuracy**: 90.5%

Validation accuracy over epochs:

![Validation Accuracy](./images/validation_accuracy.png)

Example predictions on test images:

![Predictions](./images/predictions.png)
