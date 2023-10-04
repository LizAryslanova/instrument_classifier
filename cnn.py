import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

'''
    ===================================
    CNN model
    ===================================
'''

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        kernel_1 = 5
        stride_1 = 0
        padding_1 = 0

        self.conv1 = nn.Conv2d(1, 6, kernel_1, stride_1, padding_1)     # 1 input channel
        height_1, weight_1 = output_dimensions(X_test.shape[2], X_test.shape[3], padding_1, kernel_1, stride_1)

        kernel_2 = 2
        stride_2 = 2
        padding_2 = 0

        self.pool = nn.MaxPool2d(kernel_2, stride_2, padding_2)
        height_2, weight_2 = output_dimensions(height_1, weight_1, padding_2, kernel_2, stride_2)

        kernel_3 = 5
        stride_3 = 0
        padding_3 = 0

        self.conv2 = nn.Conv2d(6, 16, kernel_3, stride_3, padding_3)
        height_3, weight_3 = output_dimensions(height_2, weight_2, padding_3, kernel_3, stride_3)


        self.fc1 = nn.Linear(16 * utils.dimensions_for_linear_layer(height_3, width_3), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


