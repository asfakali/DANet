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

- Python>=3.6
- PyTorch
- NumPy
- torchvision

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/asfakali/DABNet.git
   cd DABNet
   pip install torch torchvision numpy


## Usage
2. To use the models, import them in your Python script:
   ```bash
   from model import DAB_SNet, DAB_HNet
   # Initialize the models
   dab_s_net = DAB_SNet()
   dab_h_net = DAB_HNet()


## Training
3. To train the models, you need to set up a training loop. Here’s a basic example:
   ```bash
   import torch
   import torch.optim as optim
   from model import DAB_SNet  # or DAB_HNet


   # Initialize model, loss function, and optimizer
   model = DAB_SNet()  # or DAB_HNet
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)

   # Assume dataloader is predefined
   num_epochs = 10
   for epoch in range(num_epochs):
       model.train()  # Set model to training mode
       for inputs, labels in dataloader:
           optimizer.zero_grad()
           outputs = model(inputs)
           loss = criterion(outputs, labels)
           loss.backward()
           optimizer.step()
           
       print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

## Training Script
4. To train the model, use the `train.py` script. You need to provide several arguments:
   ```bash
   python train.py --data_dir <path_to_your_data> --batch_size <batch_size> --epochs <number_of_epochs> --lr <learning_rate> --model <DAB_SNet or DAB_HNet>


   Example
   python train.py --data_dir ./data --batch_size 32 --epochs 20 --lr 0.001 --model DAB_HNet


## Testing
5. To test the models, load a pre-trained model and evaluate it on a test dataset:
   ```bash
   # Load your model and set it to evaluation mode
   model.eval()
   
   # Assume test_loader is predefined
   correct = 0
   total = 0
   with torch.no_grad():
       for inputs, labels in test_loader:
           outputs = model(inputs)
           _, predicted = torch.max(outputs.data, 1)
           total += labels.size(0)
           correct += (predicted == labels).sum().item()
   
   print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')
   

## Testing Script
6. To evaluate the trained model, use the `test.py` script with the following arguments:
   ```bash
   python test.py --data_dir <path_to_your_data> --batch_size <batch_size> --checkpoint <path_to_model_checkpoint> --model <DAB_SNet or DAB_HNet>

   Example
   python test.py --data_dir ./data --batch_size 32 --checkpoint ./checkpoints/model.pth --model DAB_HNet


#### Arguments:
   - `--data_dir`: (required) Path to the directory containing the dataset.
   - `--batch_size`: (optional) Batch size for testing (default: 64).
   - `--checkpoint`: (required) Path to the model checkpoint file.
   - `--model`: (required) The model to test (DAB_SNet or DAB_HNet).

