# DANet: Lightweight Dilated Attention Network for Malaria Parasite Detection

## Introduction

DANet is a novel convolutional neural network designed for the accurate detection of malaria parasites in red blood cell smear images. Utilizing an innovative Dilated Attention mechanism, DANet effectively highlights critical features while maintaining a lightweight architecture with only 2.3 million parameters. Achieving an impressive accuracy of 98.02% on the NIH Malaria Cell Images Dataset, DANet offers a significant improvement in automated malaria diagnosis, making it ideal for implementation in resource-constrained environments. The models are designed to be lightweight and efficient, suitable for deployment on edge devices.

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



## Requirements

To run the models, you will need the following packages:

- Python>=3.6
- PyTorch
- NumPy
- torchvision

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/asfakali/DANet.git
   cd DANet
   pip install torch torchvision numpy


## Usage
2. To use the models, import them in your Python script:
   ```bash
   from model import DA_SNet, DA_HNet
   # Initialize the models
   da_s_net = DA_SNet()
   da_h_net = DA_HNet()


## Training
3. To train the models, you need to set up a training loop. Here’s a basic example:
   ```bash
   import torch
   import torch.optim as optim
   from model import DA_SNet  # or DB_HNet


   # Initialize model, loss function, and optimizer
   model = DA_SNet()  # or DA_HNet
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
   python train.py --data_dir <path_to_your_data> --batch_size <batch_size> --epochs <number_of_epochs> --lr <learning_rate> --model <DA_SNet or DA_HNet> --load_model <Path to pre-trained model>


##### Example
   ```python train.py --data_dir ./data --batch_size 32 --epochs 20 --lr 0.001 --model DAB_HNet```

##### Arguments:
   - `--data_dir`: (required) Path to the directory containing the dataset.
   - `--batch_size`: (optional) Batch size for testing (default: 64).
   - `--checkpoint`: (required) Path to the model checkpoint file.
   - `--lr`: (optional) Learning rate (default: 0.001).
   - `--model`: (required) The model to test (DA_SNet or DA_HNet).
   -  `--load_model`: (optional) Path to load a pre-trained model.
     
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

##### Example
   ```python test.py --data_dir ./data --batch_size 32 --checkpoint ./checkpoints/model.pth --model DAB_HNet```


##### Arguments:
   - `--data_dir`: (required) Path to the directory containing the dataset.
   - `--batch_size`: (optional) Batch size for testing (default: 64).
   - `--checkpoint`: (required) Path to the model checkpoint file.
   - `--model`: (required) The model to test (DA_SNet or DA_HNet).

