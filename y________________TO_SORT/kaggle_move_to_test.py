
from PIL import Image
from numpy import asarray
import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np
import pandas as pd

import utils

current_dir = os.path.dirname(os.path.realpath(__file__))



kaggle_train = current_dir + '/audio_files/from_kaggle/Train_submission/Train_submission/'
kaggle_test = current_dir + '/audio_files/from_kaggle/Test_submission/Test_submission/'

kaggle_train_csv = current_dir + '/audio_files/from_kaggle/Metadata_Train.csv'
kaggle_test_csv = current_dir + '/audio_files/from_kaggle/Metadata_Test.csv'


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
        else:
            label = Skip # in case the file is not in scv

    return label





import os

source = kaggle_train
destination = kaggle_test






allfiles = os.listdir(source)

# iterate on all files to move them to destination folder

guitar_n = 0
violin_n = 0
piano_n = 0
drums_n = 0
n = 10

for f in allfiles:
    instrument = 'skip'

    if get_label(f, kaggle_train_csv) == 0 and guitar_n < n:
        instrument = 'guitar'
        guitar_n += 1
    elif get_label(f, kaggle_test_csv) == 1 and piano_n < n:
        instrument = 'piano'
        piano_n +=1
    elif get_label(f, kaggle_train_csv) == 2 and drums_n < n:
        instrument = 'drums'
        drums_n += 1
    elif get_label(f, kaggle_test_csv) == 3 and violin_n < n:
        instrument = 'violin'
        violin_n += 1


    if instrument != 'skip':
        src_path = os.path.join(source, f)
        dst_path = os.path.join(destination, instrument + '_____' + f)
        os.rename(src_path, dst_path)

