
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import pickle
import audio_to_spectral_data


import os
current_dir = os.path.dirname(os.path.realpath(__file__))




def output_dimensions(width, height, padding, kernel, stride):
    '''
        ===================================
        Takes in dimensions of the input numpy array .shape[2] and .shape[3], padding, kernel and stride
            - calculates the dimentions of the output after a filter
            - returns hight and width output sizes
        ===================================
    '''
    output_width = int( ( width + 2 * padding - kernel ) / stride ) + 1
    output_height = int( ( height + 2 * padding - kernel ) / stride ) + 1

    return output_width, output_height



# Calculate dimensions for the nn.Linear layer
def dimensions_for_linear_layer(width, height):
    '''
        ===================================
        Takes in dimensions of the input numpy array (.shape[2] and .shape[3])
            - calculates the dimentions for the nn.Linear input
            - returns a product of hight and width output sizes
        ===================================
    '''

    return int(width * height)



def get_labels_from_nsynth():
    '''  Returns a list of all labels from nsynth sataset '''
    import os
    nsynth_labels = []
    big_folder = current_dir + '/audio_files/nsynth/Training_audios/'

    for i in range(31):
        folder_address = big_folder + str(i+1) + '/'
        for file in os.listdir(folder_address):
            if file[-4:] == '.wav':
                label = file[:-16]
                if label not in nsynth_labels:
                    nsynth_labels.append(label)
    return nsynth_labels



def get_classes(address = 'model.yml'):
    """ Returs tuple of classes that are in the .yml file with model parameters """
    import yaml
    with open(address, 'r') as file:
        yaml_input = yaml.safe_load(file)

    classes = ()
    for i in range(yaml_input['model_parameters']['num_classes']):
        classes = classes + (yaml_input['train_loop']['classes']['c_'+str(i+1)], )
    return classes




def look_at_mel_spectrograms(source_address = '/audio_files/from_kaggle/Test_submission/Test_submission/', destination_address = '/unit_testing/spectrograms/mel_4000/'):
    """ Takes all files in the source_address, creates a mel spectrogram for each and saves them in the destination_address.
    """
    import utils
    import os
    current_dir = os.path.dirname(os.path.realpath(__file__))

    for file in os.listdir(folder_address):
        utils.audio_to_mel_spectrogram(current_dir + source_address, file, current_dir +destination_address)




def check_quantity_of_data(name_of_yml = 'model.yml'):

    import numpy as np
    import pickle
    import utils
    import os
    import yaml

    current_dir = os.path.dirname(os.path.realpath(__file__))

    with open(name_of_yml, 'r') as file:
       yaml_input = yaml.safe_load(file)

    with open(current_dir + yaml_input['train_loop']['x_test_address'], 'rb') as f:
        X_test = pickle.load(f)
    with open(current_dir + yaml_input['train_loop']['x_train_address'], 'rb') as f:
        X_train = pickle.load(f)

    train_size = np.shape(X_train)
    test_size = np.shape(X_test)

    print('Train = ', train_size)
    print('Test = ', test_size)

