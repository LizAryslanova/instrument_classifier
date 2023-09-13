
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim

import pickle






# Calculate dimensions for the nn.Linear layer
def dimensions_for_linear_layer(width, height):
    '''
    Takes in dimensions of the input numpy array .shape[2] and .shape[3]
        - calculates the dimentions for the nn.Linear input
        - returns a product of hight and width output sizes

    '''

    # add stride, padding and kernel as input parameters!!!!!!

    o_01 = (width - 5 + 0) / 1 + 1
    o_02 = o_01 // 2
    o_03 = (o_02 - 5 + 0) / 1 + 1
    output_shape_1 = o_03 // 2

    o_11 = (height - 5 + 0) / 1 + 1
    o_12 = o_11 // 2
    o_13 = (o_12 - 5 + 0) / 1 + 1
    output_shape_2  = o_13 // 2
    # print(output_shape_1, output_shape_2)

    return int(output_shape_1 * output_shape_2)
