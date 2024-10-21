import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

def get_data_loaders(data_dir, batch_size=256, val_size=0.1, test_size=0.1):
    jitter_image = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data = datasets.ImageFolder(data_dir, transform=jitter_image)
    ntrain = len(train_data)
    indices = np.arange(ntrain)
    np.random.shuffle(indices)

    val_split = int(val_size * ntrain)
    test_split = int((val_size + test_size) * ntrain)

    train_idx = indices[test_split:]
    val_idx = indices[:val_split]
    test_idx = indices[val_split:test_split]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    params = {'batch_size': batch_size, 'num_workers': 4}
    train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, **params)
    val_loader = torch.utils.data.DataLoader(train_data, sampler=val_sampler, **params)
    test_loader = torch.utils.data.DataLoader(train_data, sampler=test_sampler, **params)

    return train_loader, val_loader, test_loader
