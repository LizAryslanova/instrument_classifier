
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

    =================================================================
    =================================================================

        Currently Im loading everything into a numpy array. Check if pytorch/tensor actually work with numpy or do they need an image with 3 channels. In that case - rewrite a thing and dont forget that my current spectrograms have 4 channels!!!!

    =================================================================
    =================================================================


'''



'''
    Function to import a RGBA .png image as a RGB numpy array
'''
def import_image(address):
    rgba_image = Image.open(address)
    rgb_image = rgba_image.convert('RGB')
    numpydata = asarray(rgb_image)
    return numpydata


'''
    Function to get the label for a file
'''
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
            label  = Skip

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
    Function to get train_x and train_y from a folder and a csv
'''



def process_image_folder(folder_address, csv_address):
    '''
        Takes in the address of a folder, converts all .wav files into spectrograms (cuts the silence, takes the first 3 seconds). Saves the spectrograms in destination_address with the same names as original .wav (but in .png)
    '''


    # how many files have a label in the csv?


    file_number = 0

    for file in os.listdir(folder_address):
        if(file[-4:] == '.png'):
            if get_label(file, csv_address) != 'skip':
                file_number += 1

                # geting the shape of the images
                if file_number < 2:
                    image = import_image(folder_address + file)
                    image_shape_a, image_shape_b, channels = image.shape

    print('Number of labelled files = ', file_number)


    # create x and y numpy arrays of correct sizes
    x = np.ndarray(shape=(file_number, image_shape_a, image_shape_b, channels),
                     dtype=np.float32)
    # y = np.ndarray(shape=(1, file_number))
    y = np.ndarray(shape=(file_number))


    # for loop that goes through all files in the folder
    count = 0
    for file in os.listdir(folder_address):
        if(file[-4:] == '.png'):
            if get_label(file, csv_address) != 'skip':

                # add a picture to train_x
                # print ('Processing: ' + str(count+1))
                x[count] = import_image(folder_address + file)

                # add a label to train_y (that corresponds to the train_x entry)
                y[count] = get_label(file, csv_address)

                count += 1
    return np.rollaxis(x, 3, 1), y

'''

kaggle_folder_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'

kaggle_csv_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'

train_x, train_y = process_image_folder(kaggle_folder_address, kaggle_csv_address)


print('Train x shape = ', train_x.shape)
print('Train y shape = ', train_y.shape)



for i in range(245):
    print('i: ',  train_y[(5*i):(5*i+5)])

'''


'''

kaggle_test_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Test_spectrograms/'

kaggle_csv_test_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Test.csv'


test_x, test_y = process_image_folder(kaggle_test_address, kaggle_csv_test_address)


print('Test x shape = ', test_x.shape)
print('Test y shape = ', test_y.shape)
'''