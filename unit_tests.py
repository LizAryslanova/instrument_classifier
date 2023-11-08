import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

import utils
from cnn import CNN

import os



current_dir = os.path.abspath(os.getcwd())







# ===========================================

classes = utils.get_classes()

# Un-Pickle Test sets
with open(current_dir + '/pickles/kaggle_test_mel_8000_no_split_test__x', 'rb') as f:
    X_test = pickle.load(f)
with open(current_dir+ '/pickles/kaggle_test_mel_8000_no_split_test__y', 'rb') as f:
    y_test = pickle.load(f)

X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

model = torch.load(current_dir + '/unit_testing/cnn_for_tests.pt')

# ===========================================

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
        print('✓ utils.test is all good')
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


# ===========================================

# utils_dimensions_for_linear_layer test
def utils_dimensions_for_linear_layer():
    '''
        Checks if calculating dimensions for the linear layer works correctly against manually calculated results
    '''
    print('===============================')
    print('Checking utils.dimensions_for_linear_layer')

    if utils.dimensions_for_linear_layer(0, 0) == 0 and utils.dimensions_for_linear_layer(100, 80) == 8000 and utils.dimensions_for_linear_layer(20, 80) == 1600:
        print('✓ utils.dimensions_for_linear_layer is all good')

    else:
        if utils.dimensions_for_linear_layer(0, 0) != 0:
            print('Correct answer to (0,0) input = 0')
            print('Function output = ', utils.dimensions_for_linear_layer(0, 0))
        if utils.dimensions_for_linear_layer(100, 80) != 8000:
            print('Correct answer to (100,80) input = 8000')
            print('Function output = ', utils.dimensions_for_linear_layer(100, 80))
        if utils.dimensions_for_linear_layer(20, 80) != 1600:
            print('Correct answer to (20,80) input = 1600')
            print('Function output = ', utils.dimensions_for_linear_layer(20, 80))

    print(' ')


# ===========================================

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
        print('✓ utils.confusion_matrix is all good')
    else:
        print('Confusion matrises dont match:')
        print('Correct = ', correct_confusion)
        print('utils.test = ', confusion)

    print(' ')


# ===========================================

# utils.plot_image test
def utils_plot_image_test():
    '''
        Runs through utils.plot_image and saves the resulting image to unit_testing folder.
        Currently need to check the image manually
    '''
    print('===============================')
    print('Checking utils.plot_image')

    num_epochs = 18
    learning_rate = 0.00002

    y_true = [1,0,0,0,0,0,1,0,0]
    y_predicted = [1,0,2,0,3,2,1,0,1]

    y_1 = [3,13,23,4,2,1,34,33]
    y_2 = [10,3,43,41,2,10,4,3]

    classes = ('Guitar', 'Piano', 'Drum', 'Violin')
    accuracies = [87.3434234234, 12.4244444, 43.00004044, 55.444434, 66.4342345]
    filename = current_dir + '/unit_testing/test.pt'
    utils.plot_image(y_1, y_2, num_epochs, learning_rate, classes, accuracies, y_true, y_predicted, filename, show = False)

    print('Check unit_testing folder')
    print(' ')


# ===========================================

def utils_get_labels_from_nsynth_test():
    print('===============================')
    print('Checking utils.get_labels_from_nsynth')

    correct_labels = ['bass_synthetic', 'mallet_acoustic', 'keyboard_electronic', 'organ_electronic', 'guitar_electronic', 'synth_lead_synthetic', 'brass_acoustic', 'flute_acoustic', 'guitar_acoustic', 'keyboard_acoustic', 'vocal_acoustic', 'guitar_synthetic', 'reed_acoustic', 'bass_electronic', 'mallet_electronic', 'string_acoustic', 'vocal_synthetic', 'brass_electronic', 'keyboard_synthetic', 'flute_synthetic', 'mallet_synthetic', 'reed_synthetic', 'bass_acoustic', 'string_electronic', 'organ_acoustic', 'vocal_electronic', 'reed_electronic', 'flute_electronic']

    function_labels = utils.get_labels_from_nsynth()

    if len(function_labels) == len(correct_labels):
        print('✓ utils.get_labels_from_nsynth is all good')

    else:
        print('Correct output = ', correct_labels)
        print(len(correct_labels))
        print('Function output = ', function_labels)
        print(len(function_labels))
    print(' ')


# ===========================================

def utils_audio_to_numpy_test():
    print('===============================')
    print('Checking utils.audio_to_numpy')
    correct_array = np.array([[10.500757], [14.129261] ,  [2.5759351], [23.461042] ])
    samples, sr = utils.audio_to_samples(current_dir + '/unit_testing/', 'G53-71607-1111-229.wav')
    function_array = utils.audio_to_numpy(samples, sr, 11025)[0, 0:4]

    if correct_array.all() == function_array.all():
        print('✓ utils.audio_to_numpy is all good')
    else:
        print('Correct output = ', correct_array)
        print('Function output = ', function_array)
    print(' ')


# ===========================================

def utils_audio_to_spectrogram_test():
    print('===============================')
    print('Checking utils.audio_to_spectrogram')

    folder_address = current_dir + '/unit_testing/'
    file = 'G53-71607-1111-229.wav'
    destination_address = current_dir + '/unit_testing/'
    utils.audio_to_spectrogram(folder_address, file, destination_address)

    print('Check unit_testing folder')
    print('')

# ===========================================

def utils_dim_of_spectrogram_test():
    print('===============================')
    print('Checking utils.dim_of_spectrogram')

    correct_dims = (1025, 130, 1)
    function_dims = utils.dim_of_spectrogram()

    if correct_dims == function_dims:
        print('✓  utils.dim_of_spectrogram is all good')
    else:
        print('Correct oupput = ', correct_dims)
        print('Function output = ', function_dims)
    print(' ')




def test_utils_output_dimensions():
    print('===============================')
    print('Checking utils.output_dimensions')

    layer1 = 6
    padding1 = 1
    stride1 = 1
    kernel1 = 3
    correct_output1 = 6
    function_output1, _ = utils.output_dimensions(layer1, layer1, padding1, kernel1, stride1)

    layer2 = 7
    padding2 = 0
    stride2 = 2
    kernel2 = 3
    correct_output2 = 3
    function_output2, _ = utils.output_dimensions(layer2, layer2, padding2, kernel2, stride2)

    if correct_output1 == function_output1 and correct_output2 == function_output2:
        print('✓  utils.output_dimension is all good')
    else:
        print('Correct output 1 = ', correct_output1)
        print('Function output 1 = ', function_output1)
        print('Correct output 2 = ', correct_output2)
        print('Function output 2 = ', function_output2)
    print(' ')




def test_utils_get_classes():
    print('===============================')
    print('Checking utils.get_classes')
    classes = utils.get_classes()
    print(classes)
    print(' ')




'''
    =================================
    Running all the tests
    =================================
'''

# utils_test()
utils_dimensions_for_linear_layer()
#confusion_matrix_test()
#utils_plot_image_test()
utils_get_labels_from_nsynth_test()
utils_audio_to_numpy_test()
utils_audio_to_spectrogram_test()
utils_dim_of_spectrogram_test()
test_utils_output_dimensions()
test_utils_get_classes()