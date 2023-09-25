import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np

import utils
import pickle


nsynth_labels = ['bass_synthetic',
                'mallet_acoustic',
                'keyboard_electronic',
                'organ_electronic',
                'guitar_electronic',
                'synth_lead_synthetic',
                'brass_acoustic',
                'flute_acoustic',
                'guitar_acoustic',
                'keyboard_acoustic',
                'vocal_acoustic',
                'guitar_synthetic',
                'reed_acoustic',
                'bass_electronic',
                'mallet_electronic',
                'string_acoustic',
                'vocal_synthetic',
                'brass_electronic',
                'keyboard_synthetic',
                'flute_synthetic',
                'mallet_synthetic',
                'reed_synthetic',
                'bass_acoustic',
                'string_electronic',
                'organ_acoustic',
                'vocal_electronic',
                'reed_electronic',
                'flute_electronic']


'''
Thoughts:

what do i want right now?

guitar: 'guitar_acoustic',
piano: keyboard_acoustic',

=====
Next ?
bass: 'bass_acoustic', 'bass_electronic', 'bass_synthetic'
vocals: 'vocal_electronic', 'vocal_synthetic', 'vocal_acoustic'

'''



def move_600_guitar_and_piano_for_transfer_learning():
    '''
    Moves 600 of each (guitar and piano labelled files) to a separate folder to be later processed for transfer learning
    count_gutar, count_piano = (11380, 8068)
    '''
    import os
    count_gutar = 0
    count_piano = 0
    big_folder = '/Users/cookie/dev/instrumant_classifier/audio_files/nsynth/Training_audios/'
    destination_address = '/Users/cookie/dev/instrumant_classifier/audio_files/nsynth/For_transfer_guitar_piano/'

    for i in range(31):
        folder_address = big_folder + str(i+1) + '/'
        for file in os.listdir(folder_address):
            if file[-4:] == '.wav':
                label = file[:-16]
                if label == 'guitar_acoustic':
                    count_gutar += 1
                    if count_gutar < 600:
                        # move that guitaf file into a special folder
                        src_path = os.path.join(folder_address, file)
                        dst_path = os.path.join(destination_address, file)
                        os.rename(src_path, dst_path)

                elif label == 'keyboard_acoustic':
                    count_piano += 1
                    if count_piano < 600:
                        # move the file on the special folder
                        src_path = os.path.join(folder_address, file)
                        dst_path = os.path.join(destination_address, file)
                        os.rename(src_path, dst_path)




# works ringt now for the folder with guitars and pianos (acoustic)
def nsynth_label(file):
    Sound_Guitar = 0
    Sound_Piano = 1
    if file[-4:] == '.wav':
        nsynth_label = file[:-16]
        if 'guitar_acoustic' in nsynth_label:
            label = Sound_Guitar
        elif 'keyboard_acoustic' in nsynth_label:
            label = Sound_Piano
        else:
            label = 'skip'
    else:
        label = 'skip'
    return label






'''
    ===================================
    Processing the folder
    ===================================
'''

def process_folder(folder_address, number_of_files):
    '''
        Takes in the address of a folder, converts all .wav files into spectrograms (cuts the silence, takes the first 3 seconds).
        Returns STFT absolute values and labels as numpy arrays
    '''

    image_shape_a, image_shape_b, channels = utils.dim_of_spectrogram()

    # create empty X and y arrays
    X_long = np.zeros(shape=(number_of_files, image_shape_a, image_shape_b, channels),
                      dtype=np.float32)
    y_long = np.zeros(shape=(number_of_files))

    # =====================
    number_of_labelled_files = 0
    for file in os.listdir(folder_address):
        if nsynth_label(file) != 'skip':      # checking if label exists for this file

            if ( number_of_labelled_files % 50 ) == 0:
                print ('Processing: ' + str(number_of_labelled_files + 1) + '   Name: ' + file)
            # !!!!! create a numpy array od the correct shape and a second one with labels !!!!!!
            X_long[number_of_labelled_files] = utils.audio_to_numpy(folder_address, file)
            y_long[number_of_labelled_files] = nsynth_label(file)
            number_of_labelled_files += 1

    # create new arrays considering the number_of_labelled_files
    X = np.zeros(shape=(number_of_labelled_files, image_shape_a, image_shape_b, channels),
                     dtype=np.float32)
    y = np.zeros(shape=(number_of_labelled_files))

    # copy only files that were labelled
    X = X_long[:number_of_labelled_files]
    y = y_long[:number_of_labelled_files]

    return np.rollaxis(X, 3, 1), y



folder_address = '/Users/cookie/dev/instrumant_classifier/audio_files/nsynth/For_transfer_guitar_piano/'
number_of_files = 1300



transfer_X, transfer_y = process_folder(folder_address, number_of_files)


print(transfer_X.shape)

with open('/Users/cookie/dev/instrumant_classifier/pickles/nsynth_transfer_x', 'wb') as f:
    pickle.dump(transfer_X , f)
with open('/Users/cookie/dev/instrumant_classifier/pickles/nsynth_transfer_y', 'wb') as f:
    pickle.dump(transfer_y , f)