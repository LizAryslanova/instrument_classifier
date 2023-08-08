
from PIL import Image
from numpy import asarray
import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np




'''
    Function to import a RGBA .png image as a RGB nupy array
'''

def import_image(address):

    if(address[-4:] == '.png'):

        rgba_image = Image.open(address)
        rgb_image = rgba_image.convert('RGB')

            # summarize some details about the image
        # print(rgb_image.format)
        # print(rgb_image.size)
        # print(rgb_image.mode)


            # asarray() class is used to convert PIL images into NumPy arrays
        numpydata = asarray(rgb_image)

            # <class 'numpy.ndarray'>
        # print(type(numpydata))
            #  shape
        # print(numpydata.shape)

        return numpydata






def process_image_folder(folder_address):
    '''
        Takes in the address of a folder, converts all .wav files into spectrograms (cuts the silence, takes the first 3 seconds). Saves the spectrograms in destination_address with the same names as original .wav (but in .png)
    '''

    number_of_train_files = 0
    channels = 3

    for file in os.listdir(folder_address):
        if file[-4:] == '.png':
            number_of_train_files += 1

            if number_of_train_files < 2:
                image = import_image(folder_address + file)
                image_shape_a, image_shape_b, channels = image.shape



    train_x = np.ndarray(shape=(number_of_train_files, image_shape_a, image_shape_b, channels),
                     dtype=np.float32)
    count = 0

    for file in os.listdir(folder_address):

        if file[-4:] == '.png':
            print ('Processing: ' + str(count+1) + '   Name: ' + file)
            train_x[count] = import_image(folder_address + file)
            count += 1

    return train_x




kaggle_folder_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'

x = process_image_folder(kaggle_folder_address)


# <class 'numpy.ndarray'>
print(type(x))
            #  shape
print(x.shape)


print(x)