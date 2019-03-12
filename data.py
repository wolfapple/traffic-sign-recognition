import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, sampler
from transform import get_train_transforms, get_test_transforms

# To load the picked dataset


class PickledDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, mode='rb') as f:
            data = pickle.load(f)
            self.features = data['features']
            self.labels = data['labels']
            self.count = len(self.labels)
            self.transform = transform

    def __getitem__(self, index):
        feature = self.features[index]
        if self.transform is not None:
            feature = self.transform(feature)
        return (feature, self.labels[index])

    def __len__(self):
        return self.count

# To move batches to the GPU


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def get_train_loaders(path, device, batch_size, workers, gray=True, augmentation=True):
    def preprocess(x, y):
        return x.to(device), y.to(device, dtype=torch.int64)

    train_dataset = PickledDataset(
        path+'/train.p', transform=get_train_transforms(gray, augmentation))
    valid_dataset = PickledDataset(
        path+'/valid.p', transform=get_test_transforms(gray))

    if augmentation:
        # Use weighted sampler
        class_sample_count = np.bincount(train_dataset.labels)
        weights = 1 / np.array([class_sample_count[y]
                                for y in train_dataset.labels])
        samp = sampler.WeightedRandomSampler(weights, 43 * 5000)
        train_loader = WrappedDataLoader(DataLoader(
            train_dataset, batch_size=batch_size, sampler=samp, num_workers=workers), preprocess)
    else:
        train_loader = WrappedDataLoader(DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers), preprocess)
    valid_loader = WrappedDataLoader(DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=workers), preprocess)

    return train_loader, valid_loader


def get_test_loader(path, device, gray=True):
    def preprocess(x, y):
        return x.to(device), y.to(device, dtype=torch.int64)

    test_dataset = PickledDataset(
        path+'/test.p', transform=get_test_transforms(gray))
    test_loader = WrappedDataLoader(DataLoader(
        test_dataset, batch_size=64, shuffle=False), preprocess)

    return test_loader
