import os
import torch
from torch.utils.data import Dataset
import numpy as np

class VectorDataset(Dataset):
    def __init__(
        self 
    ):
        # load training vectors
        self.data_array = np.load("A_matrix_8_8_8__training.npy")


    def __len__(self):
        return self.data_array.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

    
        target_vector = self.data_array[idx, :][np.newaxis, ...]
        return target_vector
