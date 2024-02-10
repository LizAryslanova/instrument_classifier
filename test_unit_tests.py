import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
import utils
from cnn import CNN
#import pytest

import plotting_results



# ===========================================

classes = utils.get_classes()

# Un-Pickle Test sets
with open(current_dir + '/pickles/kaggle_test_mel_8000_no_split_test__x', 'rb') as f:
    X_test = pickle.load(f)
with open(current_dir+ '/pickles/kaggle_test_mel_8000_no_split_test__y', 'rb') as f:
    y_test = pickle.load(f)

X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

#model = torch.load(current_dir + '/unit_testing/cnn_for_tests.pt')


# ===========================================

# utils_dimensions_for_linear_layer test
def test_utils_dimensions_for_linear_layer():
    '''
        Checks if calculating dimensions for the linear layer works correctly against manually calculated results
    '''
    assert utils.dimensions_for_linear_layer(0, 0) == 0
    assert utils.dimensions_for_linear_layer(100, 80) == 8000
    assert utils.dimensions_for_linear_layer(20, 80) == 1600

# ===========================================

def test_utils_get_labels_from_nsynth():
    correct_labels = ['bass_synthetic', 'mallet_acoustic', 'keyboard_electronic', 'organ_electronic', 'guitar_electronic', 'synth_lead_synthetic', 'brass_acoustic', 'flute_acoustic', 'guitar_acoustic', 'keyboard_acoustic', 'vocal_acoustic', 'guitar_synthetic', 'reed_acoustic', 'bass_electronic', 'mallet_electronic', 'string_acoustic', 'vocal_synthetic', 'brass_electronic', 'keyboard_synthetic', 'flute_synthetic', 'mallet_synthetic', 'reed_synthetic', 'bass_acoustic', 'string_electronic', 'organ_acoustic', 'vocal_electronic', 'reed_electronic', 'flute_electronic']
    util_labels = utils.get_labels_from_nsynth()

    assert len(util_labels) == len(correct_labels)
    assert not sum([not i in util_labels for i in correct_labels])


# ===========================================

# SOMETHING IS OFF
def test_utils_audio_to_numpy():
    correct_array = np.array([[93.91], [95.91], [96.22], [100.22]])
    samples, sr = utils.audio_to_samples(current_dir + '/unit_testing/', 'G53-71607-1111-229.wav')
    function_array = utils.audio_to_numpy(samples, sr, 11025)[0, 0:4].round(2)
    assert (correct_array == function_array).all()


# ===========================================

def test_utils_dim_of_spectrogram():
    correct_dims = (512, 130, 1)
    function_dims = utils.dim_of_spectrogram()
    assert correct_dims == function_dims


# ===========================================

def test_utils_output_dimensions():

    layer1, padding1, stride1, kernel1 = 6, 1, 1, 3
    correct_output1 = 6
    function_output1, _ = utils.output_dimensions(layer1, layer1, padding1, kernel1, stride1)

    layer2, padding2, stride2, kernel2 = 7, 0, 2, 3
    correct_output2 = 3
    function_output2, _ = utils.output_dimensions(layer2, layer2, padding2, kernel2, stride2)

    assert correct_output1 == function_output1
    assert correct_output2 == function_output2

# ===========================================

def test_utils_get_classes():
    classes = utils.get_classes(current_dir + '/unit_testing/model_testing/lr_0.0001_epochs_100_20231121-222825.yml')
    assert classes == ('bass_el', 'string_ac', 'guitar_el', 'keys_ac', 'keys_el')

# ===========================================

# utils.confusion_matrix test
def test_utils_confusion_matrix():
    '''
        Checks if generating a confusion matrix works correctly
    '''
    y_pred = torch.tensor([3, 1, 2, 1, 1, 1, 3, 1, 1, 3, 1, 3, 3, 1, 1, 3, 3, 3, 2, 1, 3, 1, 2, 1,
        1, 1, 1, 1, 3, 1, 1, 2, 2, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 2,
        1, 1, 1, 3, 2, 2, 1, 1, 3, 2, 1, 1, 2, 0, 3, 1, 1, 3, 2, 2, 1, 0, 1, 2,
        3, 1])
    y_test = torch.tensor([0, 1, 2, 1, 3, 1, 3, 1, 0, 3, 1, 3, 3, 1, 1, 3, 1, 3, 2, 1, 0, 1, 2, 1,
        1, 1, 2, 1, 3, 1, 1, 2, 0, 1, 1, 2, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 2,
        1, 1, 1, 2, 1, 2, 3, 1, 3, 2, 1, 1, 2, 0, 3, 1, 0, 3, 2, 2, 1, 0, 2, 2,
        3, 1])
    correct_confusion = np.array([[ 2,  2,  1,  2], [ 0, 34,  1,  1], [ 0,  2, 11,  2], [ 0,  3,  0, 13]])
    confusion = utils.confusion_matrix(y_test, y_pred, 4)
    assert (correct_confusion == confusion).all()

# ===========================================





# utils.plot_image test
def A__utils_plot_image():
    '''
        Runs through utils.plot_image and saves the resulting image to unit_testing folder.
        Currently need to check the image manually
    '''
    print('===============================')
    print('Checking utils.plot_image')

    num_epochs = 18
    learning_rate = 0.00002

    y_true = torch.tensor([[1],[0],[0],[3],[0],[0],[1],[0],[4]]).to(device)
    y_predicted = torch.tensor([[1],[0],[2],[1],[3],[2],[1],[0],[1]]).to(device)

    y_1 = [3,13,23,4,2,1,34,33]
    y_2 = [10,3,43,41,2,10,4,3]

    classes = ('Guitar-Long', 'Piano-Tootoo', 'Drum-Tootoo', 'Violin-Tootoo', 'Blah-Tootoo')
    accuracies = [87.342, 12.424, 43.004, 55.44, 66.4345, 93.888]
    filename = current_dir + '/unit_testing/test.pt'
    plotting_results.plot_image(y_1, y_2, num_epochs, learning_rate, classes, accuracies, y_true, y_predicted, filename, show = True)

    print('Check unit_testing folder')
    print(' ')


# ===========================================

def A__utils_audio_to_spectrogram():
    print('===============================')
    print('Checking utils.audio_to_spectrogram')

    folder_address = current_dir + '/unit_testing/'
    file = 'G53-71607-1111-229.wav'
    destination_address = current_dir + '/unit_testing/'
    utils.audio_to_spectrogram(folder_address, file, destination_address)

    print('Check unit_testing folder')
    print('')



A__utils_plot_image()
#A__utils_audio_to_spectrogram()








# DOESNT WORK!!!!

# ===========================================

# utils.test
def A__utils_test():
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

    predicted, accuracies, n_class_correct, n_class_samples = utils.test(model, X_test, y_test, classes)
    predicted = predicted.numpy()

    assert (correct_predicted == predicted).all()
    assert correct_accuracies == accuracies
    assert correct_n_class_correct == n_class_correct
    assert correct_n_class_samples == n_class_samples





'''
    =================================
    Running all the tests
    =================================
'''


#test_utils_dimensions_for_linear_layer()
#test_utils_get_labels_from_nsynth()
#test_utils_audio_to_numpy()
#test_confusion_matrix()
#test_utils_dim_of_spectrogram()
#test_utils_output_dimensions()
#test_utils_get_classes()

#A__utils_plot_image()
#A__utils_audio_to_spectrogram()

#A__utils_test()

# ===========================================


