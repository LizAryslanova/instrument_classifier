import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('/Users/cookie/Desktop/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)

        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) #_samples, 1
        self.n_samples = xy.shape[0]


    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]

    def __len__(self):
        #len(dataset)
        return self.n_samples



dataset = WineDataset()
#first_data = dataset[0]
#features, labels = first_data
#print(features, labels)


dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)

'''
# there is an issue with .next() !!!!. SOLVED by changing num_workers from 2 to 0

dataiter = iter(dataloader)
data = next(dataiter)
features, labels = data

print(features, labels)
'''


# training loop

num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4) # 4 is betch size

print(total_samples, n_iterations)



for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward, backward, update
        if (i+1)%5 == 0:
            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
