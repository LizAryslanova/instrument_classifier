import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np



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




def nsynth_label_list_2():
    '''
        making list of labels from nsynth
    '''

    nsynth_train_picture_address = '/Users/cookie/dev/instrumant_classifier/audio_files/nsynth/audio/'

    label_list = []
    label_dict = {}

    for file in os.listdir(nsynth_train_picture_address):

        if file[-4:] == '.wav':
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




print('=========================')

nsynth_label_list_2()


