import utils
import os
current_dir = os.path.abspath(os.getcwd())




folder_address = current_dir + '/audio_files/from_kaggle/Test_submission/Test_submission/'

file = '100bpm-808-like-drum-loop-74838.wav'

destination_address = current_dir + '/unit_testing/spectrograms/mel_4000/'



for file in os.listdir(folder_address):
    # utils.audio_to_spectrogram(folder_address, file, destination_address)
    utils.audio_to_mel_spectrogram(folder_address, file, destination_address)

