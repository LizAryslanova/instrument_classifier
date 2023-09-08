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



    if(file[-4:] == '.wav'):

        audio_file = folder_address + file
        samples, sample_rate = librosa.load(audio_file)
        trimed_signal, _ = librosa.effects.trim(samples, top_db=15)


        sr = 22050 # sample rate, used to cut 3 seconds
        seconds_to_cut = 3
        cut_time = int(sr * seconds_to_cut)
        cut_signal = trimed_signal[0:cut_time]

        # padding with 0 for things shorter that 3 seconds
        if (len(cut_signal) < cut_time):
            cut_signal = np.pad(cut_signal, pad_width=(0, cut_time - len(cut_signal)))


        # NB! normalizing the loudness

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

    count = 0

    for file in os.listdir(folder_address):
        count += 1
        print ('Processing: ' + str(count) + '   Name: ' + file)
        audio_to_spectrogram(folder_address, file, destination_address)



"""
    ============================================
    Processing data sets
    .wav into .png
    ============================================
"""


# kaggle set

# kaggle_train_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_submission/Train_submission/'

# kaggle_train_picture_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'


# process_folder(kaggle_train_address, kaggle_train_picture_address)



kaggle_test_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Test_submission/Test_submission/'

kaggle_test_picture_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Test_spectrograms/'

process_folder(kaggle_test_address, kaggle_test_picture_address)



# Nsynth train set

nsynth_train_address = '/Users/cookie/dev/instrumant_classifier/audio_files/nsynth/Training_audios/'

nsynth_train_picture_address = '/Users/cookie/dev/instrumant_classifier/audio_files/nsynth/Nsynth_train_spectrograms/'

# process_folder(nsynth_train_address + '1' + '/', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '2', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '3', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '4', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '5', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '6', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '7', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '8', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '9', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '10', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '11', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '12', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '13', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '14', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '15', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '16', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '17', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '18', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '19', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '20', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '21', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '22', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '23', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '24', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '25', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '26', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '27', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '28', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '29', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '30', nsynth_train_picture_address)
# process_folder(nsynth_train_address + '/' + '31', nsynth_train_picture_address)
