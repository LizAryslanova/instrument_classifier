
from PIL import Image
from numpy import asarray
import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np
import pandas as pd




'''
    =============================
    Importing and Processing Kaggle dataset
    =============================
'''



kaggle_train = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'
kaggle_test = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Test_spectrograms/'

kaggle_train_csv = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'
kaggle_test_csv = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Test.csv'




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


        # NB! normalizing the loudness

        # fig = plt.figure(figsize=[1, 1])
        # ax = fig.add_subplot(111)
        # ax.axes.get_xaxis().set_visible (False)
        # ax. axes.get_yaxis().set_visible (False)
        # ax.set_frame_on(False)
        # filename = destination_address + file[:-4] + '.png'

        D = librosa.stft(cut_signal)
        S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
        # librosa.display.specshow(S_db, x_axis='time', y_axis='log')

        # plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
        # plt.close('all')

        return S_db


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
















'''
    ===============
    Processing the folder
    ===============
'''



def process_folder(folder_address, destination_address, csv_address):
    '''
        Takes in the address of a folder, converts all .wav files into spectrograms (cuts the silence, takes the first 3 seconds). Saves the spectrograms in destination_address with the same names as original .wav (but in .png)
    '''

    number_of_labelled_files = 0
    for file in os.listdir(folder_address):
        if get_label(file, csv_address) != 'skip':      # checking if label exists for this file
            number_of_labelled_files += 1

        if get_label(file, csv_address) != 'skip':
            print ('Processing: ' + str(number_of_labelled_files) + '   Name: ' + file)

            # !!!!! create a numpy array od the correct shape and a second one with labels !!!!!!
            audio_to_numpy(folder_address, file, destination_address)

