import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model import DA_SNet, DA_HNet
from dataloader import get_data_loaders
from sklearn.metrics import precision_score, recall_score, f1_score

def train(args):
    # Load data
    train_loader, val_loader, _ = get_data_loaders(args.data_dir, args.batch_size)
    
    # Select the model
    if args.model == 'DA_SNet':
        model = DA_SNet()
    elif args.model == 'DA_HNet':
        model = DA_HNet()
    else:
        raise ValueError(f"Unknown model {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load model if specified
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        print(f"Loaded model from {args.load_model}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_predictions = []

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

            # Store predictions and labels for metrics
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader)}")

        # Validate the model
        val_loss, accuracy, precision, recall, f1 = validate(model, val_loader, criterion, device)
        
        # Save model if validation loss decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_{args.model}.pth")
            print(f"Model saved as best_model_{args.model}.pth")

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Store predictions and labels for metrics
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print(f'Validation Loss: {val_loss/len(val_loader)}, Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

    return val_loss / len(val_loader), accuracy, precision, recall, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a DA_SNet or DA_HNet model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of dataset")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs to train")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--model', type=str, choices=['DA_SNet', 'DA_HNet'], required=True, help="Model to train")
    parser.add_argument('--load_model', type=str, help="Path to load a pre-trained model")

    args = parser.parse_args()
    train(args)
