import os
import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, sampler
from transform import get_train_transforms, get_test_transforms, CLAHE_GRAY
from tqdm import tqdm

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


def extend_dataset(dataset):
    X = dataset.features
    y = dataset.labels
    num_classes = 43

    X_extended = np.empty([0] + list(dataset.features.shape)
                          [1:], dtype=dataset.features.dtype)
    y_extended = np.empty([0], dtype=dataset.labels.dtype)

    horizontally_flippable = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]
    vertically_flippable = [1, 5, 12, 15, 17]
    both_flippable = [32, 40]
    cross_flippable = np.array([
        [19, 20],
        [33, 34],
        [36, 37],
        [38, 39],
        [20, 19],
        [34, 33],
        [37, 36],
        [39, 38]
    ])

    for c in range(num_classes):
        X_extended = np.append(X_extended, X[y == c], axis=0)

        if c in horizontally_flippable:
            X_extended = np.append(
                X_extended, X[y == c][:, :, ::-1, :], axis=0)
        if c in vertically_flippable:
            X_extended = np.append(
                X_extended, X[y == c][:, ::-1, :, :], axis=0)
        if c in cross_flippable[:, 0]:
            flip_c = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(
                X_extended, X[y == flip_c][:, :, ::-1, :], axis=0)
        if c in both_flippable:
            X_extended = np.append(
                X_extended, X[y == c][:, ::-1, ::-1, :], axis=0)

        y_extended = np.append(y_extended, np.full(
            X_extended.shape[0]-y_extended.shape[0], c, dtype=y_extended.dtype))

    dataset.features = X_extended
    dataset.labels = y_extended
    dataset.count = len(y_extended)

    return dataset


def preprocess(path):
    if not os.path.exists(f"{path}/train_gray.p"):
        for dataset in ['train', 'valid', 'test']:
            with open(f"{path}/{dataset}.p", mode='rb') as f:
                data = pickle.load(f)
                X = data['features']
                y = data['labels']

            clahe = CLAHE_GRAY()
            for i in tqdm(range(len(X)), desc=f"Processing {dataset} dataset"):
                X[i] = clahe(X[i])

            X = X[:, :, :, 0]
            with open(f"{path}/{dataset}_gray.p", "wb") as f:
                pickle.dump({"features": X.reshape(
                    X.shape + (1,)), "labels": y}, f)


def get_train_loaders(path, device, batch_size, workers, class_count):
    def to_device(x, y):
        return x.to(device), y.to(device, dtype=torch.int64)

    train_dataset = extend_dataset(PickledDataset(
        path+'/train_gray.p', transform=get_train_transforms()))
    valid_dataset = PickledDataset(
        path+'/valid_gray.p', transform=get_test_transforms())

    # Use weighted sampler
    class_sample_count = np.bincount(train_dataset.labels)
    weights = 1 / np.array([class_sample_count[y]
                            for y in train_dataset.labels])
    samp = sampler.WeightedRandomSampler(weights, 43 * class_count)
    train_loader = WrappedDataLoader(DataLoader(
        train_dataset, batch_size=batch_size, sampler=samp, num_workers=workers), to_device)
    valid_loader = WrappedDataLoader(DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=workers), to_device)

    return train_loader, valid_loader


def get_test_loader(path, device, gray=True):
    def preprocess(x, y):
        return x.to(device), y.to(device, dtype=torch.int64)

    test_dataset = PickledDataset(
        path+'/test_gray.p', transform=get_test_transforms())
    test_loader = WrappedDataLoader(DataLoader(
        test_dataset, batch_size=64, shuffle=False), preprocess)

    return test_loader
