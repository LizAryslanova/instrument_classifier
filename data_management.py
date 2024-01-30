import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



class TrainSet(Dataset):
    """ Dataloader class. Need X, y at initialisation """

    def __init__(self, X_train, y_train, transform=None):
        self.x = torch.from_numpy(X_train.astype(np.float32))
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.n_samples = X_train.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        #len(dataset)
        return self.n_samples