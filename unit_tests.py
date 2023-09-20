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






# utils.test

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

    correct_accuracies = [58.108108108108105,5.555555555555555, 83.33333333333333, 66.66666666666667, 75.0]
    correct_n_class_correct = [1, 15, 12, 15]
    correct_n_class_samples = [18, 18, 18, 20]

    print('===============================')
    print('Checking utils.test')

    predicted, accuracies, n_class_correct, n_class_samples = utils.test(model, X_test, y_test, classes)
    predicted = predicted.numpy()

    if correct_predicted.all() == predicted.all() and correct_accuracies == accuracies and correct_n_class_correct == n_class_correct and correct_n_class_samples == n_class_samples:
        print('utils.test is all good')
    else:
        if correct_predicted.all() != predicted.all():
            print('Predicted tensors dont match:')
            print('Correct = ', correct_predicted)
            print('utils.test = ', predicted)
        if correct_accuracies != accuracies:
            print('Predicted accuracy doesnt match:')
            print('Correct = ', correct_acc)
            print('utils.test = ', acc)
        if correct_n_class_correct == n_class_correct:
            print('Predicted number of guesses dont match:')
            print('Correct = ', correct_n_class_correct)
            print('utils.test = ', n_class_correct)
        if correct_n_class_samples == n_class_samples:
            print('Number of examples in test set doesnt match:')
            print('Correct = ', correct_n_class_correct)
            print('utils.test = ', n_class_correct)

    print(' ')




# utils_dimensions_for_linear_layer test

def utils_dimensions_for_linear_layer():
    '''
        Checks if calculating dimensions for the linear layer works correctly against manually calculated results
    '''
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





# utils.confusion_matrix test

def confusion_matrix_test():
    '''
        Checks if generating a confusion matrix works correctly
    '''

    y_pred = torch.tensor([3, 1, 2, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 3, 3, 3, 2, 1, 3, 1, 2, 1,
        1, 1, 1, 1, 3, 1, 1, 2, 2, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 2,
        1, 1, 1, 3, 2, 2, 1, 1, 3, 2, 1, 1, 2, 0, 3, 1, 1, 3, 2, 2, 1, 0, 1, 2,
        3, 1])

    print('===============================')
    print('Checking utils.confusion_matrix')

    correct_confusion = np.array([[1, 16,  0,  1],[ 1, 15,  1,  1],[ 0,  5, 12,  1],[ 0,  5,  0, 15]])

    confusion = utils.confusion_matrix(y_test, y_pred)

    if correct_confusion.all() == confusion.all():
        print('utils.confusion_matrix is all good')
    else:
        print('Confusion matrises dont match:')
        print('Correct = ', correct_confusion)
        print('utils.test = ', confusion)

    print(' ')




def utils_plot_image_test():
    '''
        Runs through utils.plot_image and saves the resulting image to unit_testing folder.
        Currently need to check the image manually
    '''
    print('===============================')
    print('Checking utils.plot_image')



    num_epochs = 18
    learning_rate = 2.2

    y_true = [1,0,0,0,0,0,1,0,0]
    y_predicted = [1,0,2,0,3,2,1,0,1]

    y_1 = [3,13,23,4,2,1,34,33]
    y_2 = [10,3,43,41,2,10,4,3]

    classes = ('Guitar', 'Piano', 'Drum', 'Violin')
    accuracies = [87, 12, 43, 55, 66]

    destination_address = '/Users/cookie/dev/instrumant_classifier/unit_testing/'
    utils.plot_image(y_1, y_2, num_epochs, learning_rate, classes, accuracies, y_true, y_predicted, destination_address, show = False)

    print('Check unit_testing folder')
    print(' ')






confusion_matrix_test()
utils_test()
utils_dimensions_for_linear_layer()
utils_plot_image_test()