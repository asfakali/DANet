import torch
import argparse
from model import DAB_SNet, DAB_HNet
from dataloader import get_data_loaders

def test(args):
    # Load data
    _, _, test_loader = get_data_loaders(args.data_dir, args.batch_size)
    
    # Select the model
    if args.model == 'DAB_SNet':
        model = DAB_SNet()
    elif args.model == 'DAB_HNet':
        model = DAB_HNet()
    else:
        raise ValueError(f"Unknown model {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load the model checkpoint
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test a DAB_SNet or DAB_HNet model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory of dataset")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for testing")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--model', type=str, choices=['DAB_SNet', 'DAB_HNet'], required=True, help="Model to test")
    
    args = parser.parse_args()
    test(args)
