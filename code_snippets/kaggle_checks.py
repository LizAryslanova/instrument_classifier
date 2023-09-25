
from PIL import Image
from numpy import asarray
import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np
import pandas as pd




kaggle_train = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_submission/Train_submission/'
kaggle_test = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Test_submission/Test_submission/'

kaggle_train_csv = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'
kaggle_test_csv = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Test.csv'



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










import os

source = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Test_submission/Test_submission/'
destination = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/sorted_folder/'

allfiles = os.listdir(source)

# iterate on all files to move them to destination folder

for f in allfiles:
    instrument = 'skip'

    if get_label(f, kaggle_test_csv) == 0:
        instrument = 'guitar'
    elif get_label(f, kaggle_test_csv) == 1:
        instrument = 'piano'
    elif get_label(f, kaggle_test_csv) == 2:
        instrument = 'drums'
    elif get_label(f, kaggle_test_csv) == 3:
        instrument = 'violin'

    src_path = os.path.join(source, f)
    dst_path = os.path.join(destination + instrument, f)
    os.rename(src_path, dst_path)

