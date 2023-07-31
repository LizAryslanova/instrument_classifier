import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
import numpy as np

kaggle_train_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_submission/Train_submission/'

kaggle_train_picture_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'



#res = []

#for file_path in os.listdir(kaggle_train_address):
#    # check if current file_path is a file
#    if os.path.isfile(os.path.join(kaggle_train_address, file_path)):
#        # add filename to list
#        res.append(file_path[:-4])
#print(res)


# =========================================

def test_on_folder(address):

    for file in os.listdir(address):
        spectrogram_path = kaggle_train_picture_address
        audio_path = kaggle_train_address
        print ('Processing ' + file)


        audio_file = audio_path + file

        samples, sample_rate = librosa.load(audio_file, sr=None, duration=5.0)


        fig = plt.figure(figsize=[1, 1])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible (False)
        ax. axes.get_yaxis().set_visible (False)
        ax.set_frame_on(False)
        filename = spectrogram_path + file[:-4] + '.png'


        D = librosa.stft(samples)
        S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
        librosa.display.specshow(S_db, x_axis='time', y_axis='log')


        plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close('all')



test_on_folder(kaggle_train_address)
