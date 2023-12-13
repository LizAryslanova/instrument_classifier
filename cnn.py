import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml

import utils
import pickle

import os
current_dir = os.path.dirname(os.path.realpath(__file__))

with open('model.yml', 'r') as file:
    yaml_input = yaml.safe_load(file)


# Un-Pickle Test sets

with open(current_dir + yaml_input['train_loop']['x_test_address'], 'rb') as f:
    X_test = pickle.load(f)
X_test = torch.from_numpy(X_test.astype(np.float32))






'''
    ===================================
    CNN model
    ===================================
'''

class CNN(nn.Module):

    def __init__(self, model):
        super().__init__()

        kernel_conv_1 = model['kernel_conv_1']
        stride_conv_1 = model['stride_conv_1']
        padding_conv_1 = model['padding_conv_1']

        self.conv1 = nn.Conv2d(1, model['out_conv_1'], kernel_conv_1, stride_conv_1, padding_conv_1)     # 1 input channel
        height_after_conv_1, width_after_conv_1 = utils.output_dimensions(X_test.shape[2], X_test.shape[3], padding_conv_1, kernel_conv_1, stride_conv_1)

        #===========================

        kernel_pool = model['kernel_pool']
        stride_pool = model['stride_pool']
        padding_pool = model['padding_pool']

        self.pool = nn.MaxPool2d(kernel_pool, stride_pool, padding_pool)
        height_after_pool_1, width_after_pool_1 = utils.output_dimensions(height_after_conv_1, width_after_conv_1, padding_pool, kernel_pool, stride_pool)

        #===========================

        kernel_conv_2 = model['kernel_conv_2']
        stride_conv_2 = model['stride_conv_2']
        padding_conv_2 = model['padding_conv_2']

        self.conv2 = nn.Conv2d(model['out_conv_1'], model['out_conv_2'], kernel_conv_2, stride_conv_2, padding_conv_2)
        height_after_conv_2, width_after_conv_2 = utils.output_dimensions(height_after_pool_1, width_after_pool_1, padding_conv_2, kernel_conv_2, stride_conv_2)
        height_after_pool_2, width_after_pool_2 = utils.output_dimensions(height_after_conv_2, width_after_conv_2, padding_pool, kernel_pool, stride_pool)

        #===========================

        kernel_conv_3 = model['kernel_conv_3']
        stride_conv_3 = model['stride_conv_3']
        padding_conv_3 = model['padding_conv_3']

        self.conv3 = nn.Conv2d(model['out_conv_2'], model['out_conv_3'], kernel_conv_3, stride_conv_3, padding_conv_3)
        height_after_conv_3, width_after_conv_3 = utils.output_dimensions(height_after_pool_2, width_after_pool_2, padding_conv_3, kernel_conv_3, stride_conv_3)
        height_after_pool_3, width_after_pool_3 = utils.output_dimensions(height_after_conv_3, width_after_conv_3, padding_pool, kernel_pool, stride_pool)

        #===========================

        self.fc1 = nn.Linear(model['out_conv_3'] * utils.dimensions_for_linear_layer(height_after_pool_3, width_after_pool_3), model['out_fc1'])
        self.fc2 = nn.Linear(model['out_fc1'], model['out_fc2'])
        self.fc3 = nn.Linear(model['out_fc2'], model['out_fc3'])
        self.fc4 = nn.Linear(model['out_fc3'], model['num_classes'])


        #===========================

        dropout_probab = model['dropout_probab']

        self.dropout = nn.Dropout(dropout_probab)

        #===========================


    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)
