import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

# Get user name
username = os.getlogin()

# Define dataset class (which extends the utils.data.Dataset module)
class custom(torch.utils.data.Dataset):
    def __init__(self, data_paths, augment=False):
        self.data_paths = data_paths
        self.augment = augment

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = np.load(self.data_paths[idx])
        name = os.path.basename(self.data_paths[idx])
        target = torch.tensor(float(name.split('_')[0]))
        input = (2.0 * (np.float32(data) / 255.0)) - 1.0
        # Augment (or just resize)
        if self.augment:
            input, target = augment(input, target)
        return input, target

# Load dataset
def prepare(dataset_folder, split):

    # Filter train and test datasets
    data_paths  = glob.glob(dataset_folder + '/*.npy')

    # Split train/test
    num_samples = len(data_paths)
    num_train = int(num_samples * split)
    num_test = num_samples - num_train
    indices = np.arange(num_samples)
    shuffled = np.random.permutation(indices).astype(int)
    train_indices = shuffled[:num_train]
    test_indices = shuffled[num_train:]

    # Bundle
    train_data = (np.array(data_paths)[train_indices])
    test_data = (np.array(data_paths)[test_indices])

    return train_data, test_data


# Augment
def augment(data, target):
    v_flip = random.randint(0, 1)
    h_flip = random.randint(0, 1)
    if(v_flip):
        data = np.flip(data, 1)
    if(h_flip):
        data = np.flip(data, 2)
    return data.copy(), target

#FIN