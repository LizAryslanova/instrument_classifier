import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np

import utils
import pickle

current_dir = os.path.dirname(os.path.realpath(__file__))



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

5 classes
800 training files each
120 test files each

    bass_electronic
    string_acoustic
    guitar_electronic
    keyboard_acoustic
    keyboard_electronic


    bass_electronic, string_acoustic, guitar_electronic, keyboard_acoustic, keyboard_electronic

so: 4000 files train
600 files test

'''


def move_num_files_each_label(num_files_to_move, label='train'):
    '''
    Moves num_files_to_move of each (bass_electronic, string_acoustic, guitar_electronic, keyboard_acoustic, keyboard_electronic labelled files) to a separate folder to be later processed for transfer learning

    label = 'train' moves them to the train folder
    label = 'test' moves them to the test folder
    '''

    import os
    count_bass_electronic = 0
    count_string_acoustic = 0
    count_guitar_electronic = 0
    count_keyboard_acoustic = 0
    count_keyboard_electronic = 0


    big_folder = current_dir + '/audio_files/nsynth/Training_audios/'
    destination_address = current_dir + '/audio_files/nsynth/for_5_class_model/' + label



    for i in range(31):
        folder_address = big_folder + str(i+1) + '/'

        for file in os.listdir(folder_address):
            if file[-4:] == '.wav':
                label = file[:-16]

                if label == 'bass_electronic':
                    count_bass_electronic += 1
                    if count_bass_electronic < num_files_to_move:
                        # move that guitaf file into a special folder
                        src_path = os.path.join(folder_address, file)
                        dst_path = os.path.join(destination_address, file)
                        os.rename(src_path, dst_path)

                elif label == 'string_acoustic':
                    count_string_acoustic += 1
                    if count_string_acoustic < num_files_to_move:
                        # move the file on the special folder
                        src_path = os.path.join(folder_address, file)
                        dst_path = os.path.join(destination_address, file)
                        os.rename(src_path, dst_path)

                elif label == 'guitar_electronic':
                    count_guitar_electronic += 1
                    if count_guitar_electronic < num_files_to_move:
                        # move the file on the special folder
                        src_path = os.path.join(folder_address, file)
                        dst_path = os.path.join(destination_address, file)
                        os.rename(src_path, dst_path)

                elif label == 'keyboard_acoustic':
                    count_keyboard_acoustic += 1
                    if count_keyboard_acoustic < num_files_to_move:
                        # move the file on the special folder
                        src_path = os.path.join(folder_address, file)
                        dst_path = os.path.join(destination_address, file)
                        os.rename(src_path, dst_path)

                elif label == 'keyboard_electronic':
                    count_keyboard_electronic += 1
                    if count_keyboard_electronic < num_files_to_move:
                        # move the file on the special folder
                        src_path = os.path.join(folder_address, file)
                        dst_path = os.path.join(destination_address, file)
                        os.rename(src_path, dst_path)



# works right now for the folder with 5 labels
def nsynth_label(file):
    bass_electronic = 0
    string_acoustic = 1
    guitar_electronic = 2
    keyboard_acoustic = 3
    keyboard_electronic = 4

    if file[-4:] == '.wav':
        nsynth_label = file[:-16]
        if 'bass_electronic' in nsynth_label:
            label = bass_electronic
        elif 'string_acoustic' in nsynth_label:
            label = string_acoustic
        elif 'guitar_electronic' in nsynth_label:
            label = guitar_electronic
        elif 'keyboard_acoustic' in nsynth_label:
            label = keyboard_acoustic
        elif 'keyboard_electronic' in nsynth_label:
            label = keyboard_electronic
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

    image_shape_a, _, channels = utils.dim_of_spectrogram()
    image_shape_b = 32

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
            samples, sr = utils.audio_to_samples(folder_address, file, sr=False)
            X_long[number_of_labelled_files] = utils.audio_to_numpy(samples, sr, 8000, seconds_to_cut=1)
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



# ==========================

    # Moving files to train and test folders

# ==========================

num_of_classes = 5
num_files_to_move_train = 0
num_files_to_move_test = int(0.2 * num_files_to_move_train)

move_num_files_each_label(num_files_to_move_train, label='train')
move_num_files_each_label(num_files_to_move_test, label='test')

print('Moved all the files')

#exit()

# ==========================

    # Processing train and test folders

# ==========================


folder_address_train = current_dir + '/audio_files/nsynth/for_5_class_model/' + 'train/'
folder_address_test = current_dir + '/audio_files/nsynth/for_5_class_model/' + 'test/'

#number_of_files_train = num_files_to_move_train * num_of_classes
#number_of_files_test = num_files_to_move_test * num_of_classes

number_of_files_train = 3995
number_of_files_test = 596


X_train, y_train = process_folder(folder_address_train, number_of_files_train)
X_test, y_test = process_folder(folder_address_test, number_of_files_test)



with open(current_dir + '/pickles/nsynth_train_x_new_sr_1sec', 'wb') as f:
    pickle.dump(X_train , f)
with open(current_dir + '/pickles/nsynth_train_y_new_sr_1sec', 'wb') as f:
    pickle.dump(y_train , f)
with open(current_dir + '/pickles/nsynth_test_x_new_sr_1sec', 'wb') as f:
    pickle.dump(X_test , f)
with open(current_dir + '/pickles/nsynth_test_y_new_sr_1sec', 'wb') as f:
    pickle.dump(y_test , f)


os.system('say "Cookie, I pickle the numpys." ')