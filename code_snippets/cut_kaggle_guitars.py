

import librosa
import soundfile as sf
import os


folder_address = '/Users/cookie/Desktop/?/guitar/'
destination_address = '/Users/cookie/Desktop/?/cut_guitar/'

for file in os.listdir(folder_address):


    if '.DS_Store' not in file:

        if (file[-4:] == '.wav'):

            audio_file = folder_address + file
            samples, sample_rate = librosa.load(audio_file)

            sr = 22050 # sample rate, used to cut 3 seconds

            if len(samples) > 25 * sample_rate:
                seconds_to_cut = 20
            else:
                seconds_to_cut = 10
            cut_time = int(sr * seconds_to_cut)
            cut_signal = samples[cut_time:]

        # sf.write(file, yt.T, sr, subtype)

        sf.write(destination_address+file, cut_signal, sample_rate)