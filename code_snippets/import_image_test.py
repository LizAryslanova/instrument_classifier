
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






def process_image_folder(folder_address):
    '''
        Takes in the address of a folder, converts all .wav files into spectrograms (cuts the silence, takes the first 3 seconds). Saves the spectrograms in destination_address with the same names as original .wav (but in .png)
    '''

    count = 0

    x = np.empty([310, 308, 3])

    for file in os.listdir(folder_address):

        while count < 20:
            count += 1
            print ('Processing: ' + str(count) + '   Name: ' + file)

            np.append(x, import_image(folder_address + file))

    return x




kaggle_folder_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'

x = process_image_folder(kaggle_folder_address)


# <class 'numpy.ndarray'>
print(type(x))
            #  shape
print(x.shape)