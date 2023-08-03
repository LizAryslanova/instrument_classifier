import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np

"""
    ============================================
    Functions for processing audio files
    ============================================
"""

def audio_to_spectrogram(folder_address, file, destination_address):
    '''
        Takes in the *file* from the *folder_address*, checks if the file is a .wav and if True:
            - trims the silence,
            - takes the first 3 seconds
            - creates a spectrogram image
            - saves image destination_address with the same name as the original file (.png)
    '''

    sr = 22050 # sample rate, used to cut 3 seconds

    if(file[-4:] == '.wav'):

        audio_file = folder_address + file
        samples, sample_rate = librosa.load(audio_file)
        trimed_signal, _ = librosa.effects.trim(samples, top_db=15)
        cut_signal = trimed_signal[0:int(sr*3)]

        fig = plt.figure(figsize=[1, 1])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible (False)
        ax. axes.get_yaxis().set_visible (False)
        ax.set_frame_on(False)
        filename = destination_address + file[:-4] + '.png'

        D = librosa.stft(cut_signal)
        S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
        librosa.display.specshow(S_db, x_axis='time', y_axis='log')

        plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close('all')




def process_folder(folder_address, destination_address):
    '''
        Takes in the address of a folder, converts all .wav files into spectrograms (cuts the silence, takes the first 3 seconds). Saves the spectrograms in destination_address with the same names as original .wav (but in .png)
    '''

    for file in os.listdir(folder_address):
        print ('Processing ' + file)
        audio_to_spectrogram(folder_address, file, destination_address)



"""
    ============================================
    Processing data sets
    .wav into .png
    ============================================
"""

kaggle_train_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_submission/Train_submission/'

kaggle_train_picture_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'


process_folder(kaggle_train_address, kaggle_train_picture_address)
