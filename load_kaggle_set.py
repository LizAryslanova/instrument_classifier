
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
    Function to import a RGBA .png image as a RGB nupy array
'''

def import_image(address):
    rgba_image = Image.open(address)
    rgb_image = rgba_image.convert('RGB')
    numpydata = asarray(rgb_image)
    return numpydata

'''
    Function to get the label for a file
'''

def get_label(file):

    Sound_Guitar = 0
    Sound_Drum = 1
    Sound_Violin = 2
    Sound_Piano = 3

    kaggle_csv_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'
    df = pd.read_csv(kaggle_csv_address)
    row_num = df[df['FileName'] == file[:-4] + '.wav'].index.to_numpy() # gets a row number of the entry (as a numpy array)

    if row_num.shape == (1,):
        # print ('Processing ' + file)
        label_str = df['Class'][row_num[0]]

        if label_str == 'Sound_Guitar':
            label = Sound_Guitar
        elif label_str == 'Sound_Drum':
            label = Sound_Drum
        elif label_str == 'Sound_Violin':
            label = Sound_Violin
        elif label_str == 'Sound_Piano':
            label = Sound_Piano

    else:
        label = 'skip' # in case the fie is not in scv

    return label



'''
    Function to get train_x and train_y from a folder and a csv
'''



def process_image_folder(folder_address):
    '''
        Takes in the address of a folder, converts all .wav files into spectrograms (cuts the silence, takes the first 3 seconds). Saves the spectrograms in destination_address with the same names as original .wav (but in .png)
    '''


    # how many files have a label in the csv?

    file_number = 0

    for file in os.listdir(folder_address):
        if(file[-4:] == '.png'):
            if get_label(file) != 'skip':
                file_number += 1

                # geting the shape of the images
                if file_number < 2:
                    image = import_image(folder_address + file)
                    image_shape_a, image_shape_b, channels = image.shape


    print('Number of labellled fies = ', file_number)


    # create train_x and train_y numpy arrays of correct sizes


    train_x = np.ndarray(shape=(file_number, image_shape_a, image_shape_b, channels),
                     dtype=np.float32)

    train_y = np.ndarray(shape=(1, file_number))


    # for loop that goes through all files in the folder

    count = 0

    for file in os.listdir(folder_address):

        if(file[-4:] == '.png'):
            if get_label(file) != 'skip':

                # add a picture to train_x
                print ('Processing: ' + str(count+1))
                train_x[count] = import_image(folder_address + file)

                # add a label to train_y (that corresponds to the train_x entry)
                train_y[0, count] = get_label(file)

                count += 1

    return train_x, train_y





kaggle_folder_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'

train_x, train_y = process_image_folder(kaggle_folder_address)


print('Train x shape = ', train_x.shape)
print('Train y shape = ', train_y.shape)


