
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




def test(CNN_model, X_test, y_test, classes):
    '''
        Runs the test sets through the model without changing the gradients.
        Prints predicted classes, accuracy of the entire network,
    '''

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        num_classes = len(classes)
        n_class_correct = [0 for i in range(num_classes)]
        n_class_samples = [0 for i in range(num_classes)]


        outputs = CNN_model(X_test)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += y_test.size(0)
        n_correct += (predicted == y_test).sum().item()

        #print('Predicted = ', predicted)

        for i in range(X_test.shape[0]):   # number of files in the test set
            label = y_test[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        # print(f'Accuracy of the network: {acc} %')

        acc_classes = []

        for i in range(num_classes):
            acc_classes.append(100.0 * n_class_correct[i] / n_class_samples[i])
            # print(f'Accuracy of {classes[i]}: {acc} %')
            # print('n_class_correct = ', n_class_correct, ' n_class_samples = ', n_class_samples)

        return predicted, acc, acc_classes, n_class_correct, n_class_samples





