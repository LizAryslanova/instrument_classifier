import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import yaml
import utils



class CNN(nn.Module):
    """ CNN model:
    conv1 -> relu -> pool ->
    conv2 -> rule -> pool ->
    conv3 -> relu -> pool ->
    fc1 -> relu -> dropout ->
    fc2 -> relu -> dropout ->
    fc3 -> relu ->
    fc4
    """

    def __init__(self, image_width, image_height, kernel_conv_1 = 2, stride_conv_1 = 1, padding_conv_1 = 2, out_conv_1 = 6, kernel_conv_2 = 3, stride_conv_2 = 1, padding_conv_2 = 2, out_conv_2 = 16, kernel_conv_3 = 5, stride_conv_3 = 1, padding_conv_3 = 2, out_conv_3 = 40, kernel_pool = 2, stride_pool = 2, padding_pool = 0, out_fc1 = 400, out_fc2 = 200, out_fc3 = 84, num_classes = 5, dropout_probab = 0.3):
        super().__init__()



        self.conv1 = nn.Conv2d(1, out_conv_1, kernel_conv_1, stride_conv_1, padding_conv_1)     # 1 input channel

        height_after_conv_1, width_after_conv_1 = utils.output_dimensions(image_width, image_height, padding_conv_1, kernel_conv_1, stride_conv_1)

        #===========================

        self.pool = nn.MaxPool2d(kernel_pool, stride_pool, padding_pool)

        height_after_pool_1, width_after_pool_1 = utils.output_dimensions(height_after_conv_1, width_after_conv_1, padding_pool, kernel_pool, stride_pool)

        #===========================

        self.conv2 = nn.Conv2d(out_conv_1, out_conv_2, kernel_conv_2, stride_conv_2, padding_conv_2)

        height_after_conv_2, width_after_conv_2 = utils.output_dimensions(height_after_pool_1, width_after_pool_1, padding_conv_2, kernel_conv_2, stride_conv_2)
        height_after_pool_2, width_after_pool_2 = utils.output_dimensions(height_after_conv_2, width_after_conv_2, padding_pool, kernel_pool, stride_pool)

        #===========================

        self.conv3 = nn.Conv2d(out_conv_2, out_conv_3, kernel_conv_3, stride_conv_3, padding_conv_3)

        height_after_conv_3, width_after_conv_3 = utils.output_dimensions(height_after_pool_2, width_after_pool_2, padding_conv_3, kernel_conv_3, stride_conv_3)
        height_after_pool_3, width_after_pool_3 = utils.output_dimensions(height_after_conv_3, width_after_conv_3, padding_pool, kernel_pool, stride_pool)

        #===========================

        self.fc1 = nn.Linear(out_conv_3 * utils.dimensions_for_linear_layer(height_after_pool_3, width_after_pool_3), out_fc1)
        self.fc2 = nn.Linear(out_fc1, out_fc2)
        self.fc3 = nn.Linear(out_fc2, out_fc3)
        self.fc4 = nn.Linear(out_fc3, num_classes)

        #===========================

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


