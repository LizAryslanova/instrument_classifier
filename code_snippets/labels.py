import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np
import pandas as pd



'''
    ======================================
    Nsynth all the labels
    ======================================

'''


def nsynth_label_list():
    '''
        making list of labels from nsynth
    '''

    nsynth_train_picture_address = '/Users/cookie/dev/instrumant_classifier/audio_files/nsynth/Nsynth_train_spectrograms/'

    label_list = []
    label_dict = {}

    for file in os.listdir(nsynth_train_picture_address):

        if file[-4:] == '.png':
            if file[:-16] not in label_list:
                label_list.append(file[:-16])
                label_dict[ file[:-16] ] = 1
            else:
                label_dict[ file[:-16] ] += 1

    All_files = 0



    for key, value in label_dict.items():
        print(key, ' : ', value)
        All_files += value

    print('All_files = ', All_files)



nsynth_label_list()


'''
    ======================================
    Kaggle all the labels
    ======================================

'''

def get_label(file):

    kaggle_csv_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'
    df = pd.read_csv(kaggle_csv_address)
    row_num = df[df['FileName'] == file[:-4] + '.wav'].index.to_numpy() # gets a row number of the entry (as a numpy array)

    if row_num.shape == (1,):
        print ('Processing ' + file)
        label = df['Class'][row_num[0]]
    else:
        label = 'skip' # in case the fie is not in scv

    return label





kaggle_train_picture_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'
counter = 0

for file in os.listdir(kaggle_train_picture_address):

    if get_label(file) != 'skip':

        counter += 1
        print('Processing # = ' + str(counter) + '    ' + file)

