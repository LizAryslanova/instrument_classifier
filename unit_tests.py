import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

import utils


'''
    ===================================
    CNN model. This is a copy-paste solution. there has to be a better option!!!!
    ===================================
'''

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)     # 1 input channel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * utils.dimensions_for_linear_layer(X_test.shape[2], X_test.shape[3]), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ===========================================

classes = ('guitar', 'piano', 'drum', 'violin')

# Un-Pickle Test sets
with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_test_x', 'rb') as f:
    X_test = pickle.load(f)
with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_test_y', 'rb') as f:
    y_test = pickle.load(f)


X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

model = torch.load('/Users/cookie/dev/instrumant_classifier/unit_testing/cnn_for_tests.pt')





'''
    utils.test
'''

def utils_test():
    '''
        Correct data input was done manually for the model that was saved to unit_testing folder.
        Loads the model state from unit_testing folder and runs it through utils.test function.
        Compares the correct data with date from the function.
    '''
    # Comparison data
    correct_predicted = torch.tensor([3, 1, 2, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 3, 3, 3, 2, 1, 3, 1, 2, 1,
        1, 1, 1, 1, 3, 1, 1, 2, 2, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 2,
        1, 1, 1, 3, 2, 2, 1, 1, 3, 2, 1, 1, 2, 0, 3, 1, 1, 3, 2, 2, 1, 0, 1, 2,
        3, 1])
    correct_predicted = correct_predicted.numpy()
    correct_acc = 58.108108108108105
    correct_acc_classes = [5.555555555555555, 83.33333333333333, 66.66666666666667, 75.0]
    correct_n_class_correct = [1, 15, 12, 15]
    correct_n_class_samples = [18, 18, 18, 20]

    print('===============================')
    print('Checking utils.test')

    predicted, acc, acc_classes, n_class_correct, n_class_samples = utils.test(model, X_test, y_test, classes)
    predicted = predicted.numpy()

    if correct_predicted.all() == predicted.all() and correct_acc == acc and correct_acc_classes == acc_classes and correct_n_class_correct == n_class_correct and correct_n_class_samples == n_class_samples:
        print('utils.test is all good')
    else:
        if correct_predicted.all() != predicted.all():
            print('Predicted tensors dont match:')
            print('Correct = ', correct_predicted)
            print('utils.test = ', predicted)
        if correct_acc != acc:
            print('Predicted accuracy doesnt match:')
            print('Correct = ', correct_acc)
            print('utils.test = ', acc)
        if correct_acc_classes == acc_classes:
            print('Predicted instrument accuracies dont match:')
            print('Correct = ', correct_acc_classes)
            print('utils.test = ', acc_classes)
        if correct_n_class_correct == n_class_correct:
            print('Predicted number of guesses dont match:')
            print('Correct = ', correct_n_class_correct)
            print('utils.test = ', n_class_correct)
        if correct_n_class_samples == n_class_samples:
            print('Number of examples in test set doesnt match:')
            print('Correct = ', correct_n_class_correct)
            print('utils.test = ', n_class_correct)

    print(' ')



def utils_dimensions_for_linear_layer():
    print('===============================')
    print('Checking utils.dimensions_for_linear_layer')

    if utils.dimensions_for_linear_layer(0, 0) == 9 and utils.dimensions_for_linear_layer(100, 80) == 374 and utils.dimensions_for_linear_layer(20, 80) == 34:
        print('utils.dimensions_for_linear_layer is all good')

    else:
        if utils.dimensions_for_linear_layer(0, 0) != 9:
            print('Correct answer to (0,0) input = 9')
            print('Function output = ', utils.dimensions_for_linear_layer(0, 0))
        if utils.dimensions_for_linear_layer(100, 80) != 374:
            print('Correct answer to (100,80) input = 374')
            print('Function output = ', utils.dimensions_for_linear_layer(100, 80))
        if utils.dimensions_for_linear_layer(20, 80) != 34:
            print('Correct answer to (20,80) input = 34')
            print('Function output = ', utils.dimensions_for_linear_layer(20, 80))

    print(' ')







utils_test()
utils_dimensions_for_linear_layer()