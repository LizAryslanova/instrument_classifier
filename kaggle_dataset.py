
from PIL import Image
from numpy import asarray
import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np
import pandas as pd

import pickle
import utils

current_dir = os.path.abspath(os.getcwd())



'''
    ===================================
    Importing and Processing Kaggle dataset
    ===================================
'''

kaggle_train = current_dir + '/audio_files/from_kaggle/Train_submission/Train_submission/'
kaggle_test = current_dir + '/audio_files/from_kaggle/Test_submission/Test_submission/'

kaggle_train_csv = current_dir + '/audio_files/from_kaggle/Metadata_Train.csv'
kaggle_test_csv = current_dir + '/audio_files/from_kaggle/Metadata_Test.csv'


number_of_files_train = 2630
number_of_files_test = 500



def get_label(file, csv_address):

    Sound_Guitar = 0
    Sound_Piano = 1
    Sound_Drum = 2
    Sound_Violin = 3
    Skip = 'skip'

    df = pd.read_csv(csv_address)
    row_num = df[df['FileName'] == file[:-4] + '.wav'].index.to_numpy() # gets a row number of the entry (as a numpy array)

    if row_num.shape != (0,):
        # print ('Processing ' + file)
        label_str = df['Class'][row_num[0]]
        if label_str == 'Sound_Guitar':
            label = Sound_Guitar
        elif label_str == 'Sound_Piano':
            label = Sound_Piano
        elif label_str == 'Sound_Drum':
            label = Sound_Drum
        elif label_str == 'Sound_Violin':
            label = Sound_Violin
        else:
            label = Skip
    else:
        label = Skip

    # Checking names of files to get the label
    if label != Sound_Guitar and label != Sound_Piano and label != Sound_Violin and label != Sound_Drum:
        if 'violin' in file or 'VIOLIN' in file or 'Vn' in file or 'Violin' in file:
            label = Sound_Violin
        elif 'guitar' in file:
            label = Sound_Guitar
        elif 'piano' in file:
            label = Sound_Piano
        elif 'drums' in file:
            label = Sound_Drum
        else:
            label = Skip # in case the file is not in scv

    return label



'''
    ===================================
    Processing the folder
    ===================================
'''

def process_folder(folder_address, csv_address, number_of_files, fmax, training = True):
    '''
        Takes in the address of a folder, converts all .wav files into spectrograms (cuts the silence, takes the first 3 seconds).
        Returns STFT absolute values and labels as numpy arrays
    '''

    image_shape_a, image_shape_b, channels = utils.dim_of_spectrogram()

    # create empty X and y arrays
    X_long = np.zeros(shape=(2* number_of_files, image_shape_a, image_shape_b, channels),
                      dtype=np.float32)
    y_long = np.zeros(shape=(2 * number_of_files))

    # =====================

    number_of_labelled_files = 0

    for file in os.listdir(folder_address):
        if get_label(file, csv_address) != 'skip':      # checking if label exists for this file

            if ( number_of_labelled_files % 50 ) == 0:
                print ('Processing: ' + str(number_of_labelled_files + 1) + '   Name: ' + file)
            # !!!!! create a numpy array od the correct shape and a second one with labels !!!!!!
            samples, sample_rate = utils.audio_to_samples(folder_address, file)
            X_long[number_of_labelled_files] = utils.audio_to_numpy(samples, sample_rate, fmax)
            y_long[number_of_labelled_files] = get_label(file, csv_address)
            number_of_labelled_files += 1




    # Looking at the length of the files. For ones that are longer than 5 seconds - cut/trim another piece to add to the training set

    if training == True:
        print('Starting a second run')

        for file in os.listdir(folder_address):
            if get_label(file, csv_address) != 'skip':      # checking if label exists for this file

                samples, sample_rate = utils.audio_to_samples(folder_address, file)
                seconds_to_cut = 5

                # check if its long enough
                if len(samples) > (seconds_to_cut + 5) * sample_rate:

                    samples, sample_rate = utils.cut_audio_to_samples(samples, sample_rate, seconds_to_cut)

                    if ( number_of_labelled_files % 50 ) == 0:
                        print ('Processing: ' + str(number_of_labelled_files + 1) + '   Name: ' + file)

                   # create a numpy array od the correct shape and a second one with labels
                    X_long[number_of_labelled_files] = utils.audio_to_numpy(samples, sample_rate, fmax)

                    y_long[number_of_labelled_files] = get_label(file, csv_address)
                    number_of_labelled_files += 1



    # create new arrays considering the number_of_labelled_files
    X = np.zeros(shape=(number_of_labelled_files, image_shape_a, image_shape_b, channels),
                     dtype=np.float32)
    y = np.zeros(shape=(number_of_labelled_files))

    # copy only files that were labelled
    X = X_long[:number_of_labelled_files]
    y = y_long[:number_of_labelled_files]

    print('Number of labelled files = ', number_of_labelled_files)

    return np.rollaxis(X, 3, 1), y



'''
    ===================================
    Running Kaggle dataset through process_folder
    ===================================
'''

fmax = 8000
notes = '_no_split_test_'

train_x, train_y = process_folder(kaggle_train, kaggle_train_csv, number_of_files_train, fmax)
test_x, test_y = process_folder(kaggle_test, kaggle_test_csv, number_of_files_test, fmax, training=False)


with open(current_dir + '/pickles/kaggle_train_mel_' + str(fmax) + notes +'_x', 'wb') as f:
    pickle.dump(train_x , f)
with open(current_dir + '/pickles/kaggle_train_mel_' + str(fmax) + notes +'_y', 'wb') as f:
    pickle.dump(train_y , f)
with open(current_dir + '/pickles/kaggle_test_mel_' + str(fmax) + notes +'_x', 'wb') as f:
    pickle.dump(test_x , f)
with open(current_dir + '/pickles/kaggle_test_mel_' + str(fmax) + notes +'_y', 'wb') as f:
    pickle.dump(test_y , f)




'''
print('Train x shape = ', train_x.shape)
print('Train y shape = ', train_y.shape)

for i in range(245):
    print('i: ',  train_y[(5*i):(5*i+5)])

print('Test x shape = ', test_x.shape)
print('Test y shape = ', test_y.shape)
'''

import os
os.system('say "Cookie, I pickled the numpys." ')