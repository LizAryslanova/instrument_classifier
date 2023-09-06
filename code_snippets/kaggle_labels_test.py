
from PIL import Image
from numpy import asarray
import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np
import pandas as pd





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
            label  = 'boo'
    else:
        label = Skip # in case the file is not in scv



    # Checking names of files to get the label
    if label != Sound_Guitar and label != Sound_Piano and label != Sound_Violin and label != Sound_Drum:
        if 'violin' in file or 'VIOLIN' in file:
            label = Sound_Violin
        else:
            label = Skip # in case the file is not in scv


    return label







kaggle_folder_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'

kaggle_csv_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'






count = 0
for file in os.listdir(kaggle_folder_address):
    if(file[-4:] == '.png'):
        if get_label(file, kaggle_csv_address) != 'skip':
            count += 1

print('COUNT = ', count)