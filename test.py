import torch
import argparse
from model import DA_SNet, DA_HNet
from dataloader import get_data_loaders
from sklearn.metrics import precision_score, recall_score, f1_score

def test(args):
    # Load data
    _, _, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    
    # Select the model
    if args.model == 'DA_SNet':
        model = DA_SNet()
    elif args.model == 'DA_HNet':
        model = DA_HNet()
    else:
        raise ValueError(f"Unknown model {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the model checkpoint
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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

    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a DA_SNet or DA_HNet model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of dataset")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for testing")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--model', type=str, choices=['DA_SNet', 'DA_HNet'], required=True, help="Model to test")
    
    args = parser.parse_args()
    test(args)
