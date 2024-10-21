import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model import DAB_SNet, DAB_HNet
from dataloader import get_data_loaders

def train(args):
    # Load data
    train_loader, val_loader, _ = get_data_loaders(args.data_dir, args.batch_size)
    
    # Select the model
    if args.model == 'DAB_SNet':
        model = DAB_SNet()
    elif args.model == 'DAB_HNet':
        model = DAB_HNet()
    else:
        raise ValueError(f"Unknown model {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader)}")
        
        # Validate the model
        validate(model, val_loader, criterion, device)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {100 * correct / total}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a DAB_SNet or DAB_HNet model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of dataset")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--model', type=str, choices=['DAB_SNet', 'DAB_HNet'], required=True, help="Model to train")
    
    args = parser.parse_args()
    train(args)
