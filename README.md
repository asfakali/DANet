# DAB_SNet and DAB_HNet Models

This repository contains implementations of the DAB_SNet and DAB_HNet models, which utilize attention mechanisms for improved performance in image classification tasks. The models are designed to be lightweight and efficient, suitable for deployment on edge devices.

## Table of Contents

- [Introduction](#introduction)
- [Model Architectures](#model-architectures)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Testing](#testing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

DAB_SNet and DAB_HNet leverage advanced convolutional techniques and attention mechanisms to enhance image classification accuracy. The models are structured to efficiently process images while minimizing computational overhead, making them ideal for resource-constrained environments. The DAB_SNet uses a traditional attention mechanism, while the DAB_HNet incorporates LazyBatchNorm and PReLU activations for improved stability during training.

## Model Architectures

### DAB_SNet

- **Architecture**: Comprises multiple convolutional layers with attention mechanisms.
- **Activation Functions**: Utilizes Leaky ReLU and Softmax for output classification.
- **Pooling**: Employs both average and max pooling to reduce dimensionality.

### DAB_HNet

- **Architecture**: Similar to DAB_SNet but employs PReLU and LazyBatchNorm for better training stability.
- **Attention Mechanism**: Implements a multi-stage attention mechanism for improved feature extraction.

## Requirements

To run the models, you will need the following packages:

- Python 3.x
- PyTorch
- NumPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
