
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



'''
    =============================
    Importing and Processing Kaggle dataset
    =============================
'''



kaggle_train = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_submission/Train_submission/'
kaggle_test = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Test_submission/Test_submission/'

kaggle_train_csv = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'
kaggle_test_csv = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Test.csv'


number_of_files_train = 2630
number_of_files_test = 80


def audio_to_numpy(folder_address, file):
    '''
        Takes in the *file* from the *folder_address*, checks if the file is a .wav and if True:
            - trims the silence,
            - takes the first 3 seconds
            - padds for files shorter than 3 seconds
            - stft
            - returns a numpy array
    '''


    if(file[-4:] == '.wav'):

        audio_file = folder_address + file
        samples, sample_rate = librosa.load(audio_file)
        trimmed_signal, _ = librosa.effects.trim(samples, top_db=15)

        sr = 22050 # sample rate, used to cut 3 seconds
        seconds_to_cut = 3
        cut_time = int(sr * seconds_to_cut)
        cut_signal = trimmed_signal[0:cut_time]

        # padding with 0 for things shorter that 3 seconds
        if (len(cut_signal) < cut_time):
            cut_signal = np.pad(cut_signal, pad_width=(0, cut_time - len(cut_signal)))

        D = librosa.stft(cut_signal)
        D_abs = np.abs(D)

        #print(D_abs.shape)
        #print(D_abs[10,50])

        a, b = D.shape
        channels = 1    # use 3 for RGB
        multiplyer = 1   # use 255 for greyscale RGB

        img = np.zeros(shape=(a, b, channels))

        #print(img[10,50,:])

        for i in range(channels):
            img[:,:,i] = multiplyer * D_abs

        #print(img[10,50,:])
        #print('img.shape= ', img.shape)

        return img



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
        if 'violin' in file or 'VIOLIN' in file:
            label = Sound_Violin
        else:
            label = Skip # in case the file is not in scv

    return label


# Shape of the spectrogram
def dim_of_spectrogram():
    folder_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_submission/Train_submission/'

    done = False

    import random
    while done == False:
        file = '0_john-garner_bwv1002_mov1.wav'
        #file = random.choice(os.listdir(folder_address)) #change dir name to whatever
        if (file[-4:] == '.wav'):
            N = audio_to_numpy(folder_address, file)
            done = True
    return N.shape





'''
    ===============
    Processing the folder
    ===============
'''

def process_folder(folder_address, csv_address, number_of_files):
    '''
        Takes in the address of a folder, converts all .wav files into spectrograms (cuts the silence, takes the first 3 seconds).
        Returns STFT absolute values and labels as numpy arrays
    '''

    image_shape_a, image_shape_b, channels = dim_of_spectrogram()

    # create empty X and y arrays
    X_long = np.zeros(shape=(number_of_files, image_shape_a, image_shape_b, channels),
                     dtype=np.float32)
    y_long = np.zeros(shape=(number_of_files))

    # =====================
    number_of_labelled_files = 0
    for file in os.listdir(folder_address):
        if get_label(file, csv_address) != 'skip':      # checking if label exists for this file

            if ( number_of_labelled_files % 50 ) == 0:
                print ('Processing: ' + str(number_of_labelled_files + 1) + '   Name: ' + file)
            # !!!!! create a numpy array od the correct shape and a second one with labels !!!!!!
            X_long[number_of_labelled_files] = audio_to_numpy(folder_address, file)
            y_long[number_of_labelled_files] = get_label(file, csv_address)
            number_of_labelled_files += 1

    # create new arrays considering the number_of_labelled_files
    X = np.zeros(shape=(number_of_labelled_files, image_shape_a, image_shape_b, channels),
                     dtype=np.float32)
    y = np.zeros(shape=(number_of_labelled_files))

    # copy only files that were labelled
    X = X_long[:number_of_labelled_files]
    y = y_long[:number_of_labelled_files]

    return np.rollaxis(X, 3, 1), y



'''
    ============================
    Running Kaggle dataset through process_folder
    ============================
'''



train_x, train_y = process_folder(kaggle_train, kaggle_train_csv, number_of_files_train)
test_x, test_y = process_folder(kaggle_test, kaggle_test_csv, number_of_files_test)


with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_train_x', 'wb') as f:
    pickle.dump(train_x , f)
with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_train_y', 'wb') as f:
    pickle.dump(train_y , f)
with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_test_x', 'wb') as f:
    pickle.dump(test_x , f)
with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_test_y', 'wb') as f:
    pickle.dump(test_y , f)




'''
print('Train x shape = ', train_x.shape)
print('Train y shape = ', train_y.shape)

for i in range(245):
    print('i: ',  train_y[(5*i):(5*i+5)])

print('Test x shape = ', test_x.shape)
print('Test y shape = ', test_y.shape)
'''


print('Im done')