



# ==================================================
# Drawing a waveform


# import scipy
# from scipy.io import wavfile

# samplerate, data = wavfile.read("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/1.wav")

# length = data.shape[0] / samplerate

# import matplotlib.pyplot as plt
import numpy as np
# time = np.linspace(0., length, data.shape[0])
# plt.plot(time, data[:, 0], label="Left channel")
# plt.plot(time, data[:, 1], label="Right channel")
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.show()





# what's the minimum sample rate that would be good enough for classification? here its 22050 Hz. it this too low? Answer: use 'sr = None' in load;  add sr=sr in specshow

import matplotlib.pyplot as plt
import librosa
import librosa.display


# # Load audio file
# audio_file = '/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/1.wav'
# y, sr = librosa.load(audio_file, mono=False, sr=None)
# y_mono = y[0,]

# # Calculate and plot spectrogram
# D = librosa.stft(y_mono)
# S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
#plt.figure(figsize=(10, 5))
# librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr)
#plt.colorbar(format='%+2.0f dB')
#plt.title('Spectrogram - Librosa')
#plt.xlabel('Time (s)')
#plt.ylabel('Frequency (Hz)')
# plt.show()



kaggle_train_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_submission/Train_submission/'



from librosa.effects import trim

# =================================

def create_spectrogram (file):
    spectrogram_path = '/Users/cookie/dev/instrumant_classifier/code_snippets/'
    audio_path = kaggle_train_address
    print ('Processing ' + file)


    audio_file = audio_path + file + '.wav'

    samples, sample_rate = librosa.load(audio_file)

    trimed_signal, _ = librosa.effects.trim(samples, top_db=15)

    print('LEN trimmed = ', len(trimed_signal))

    seconds_to_cut = 3
    sr = 20050
    cut_time = int(sr * seconds_to_cut)
    cut_signal = trimed_signal[0:cut_time]

    print('LEN cut = ', len(cut_signal))
    print('cut_time = ', cut_time)
    print('shape = ', cut_signal.shape)
    print('type = ', type(cut_signal))

    if (len(cut_signal) < cut_time):
        cut_signal = np.pad(cut_signal, pad_width=(0, cut_time - len(cut_signal)))



    fig = plt.figure(figsize=[1,1])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible (False)
    ax.axes.get_yaxis().set_visible (False)
    ax.set_frame_on(False)

    filename = spectrogram_path + 'test_image' + '.png'

    D = librosa.stft(cut_signal)
    S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
    librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sample_rate)


    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close('all')





create_spectrogram ('WaveDrum02_43HH (68)')

# ============================================


