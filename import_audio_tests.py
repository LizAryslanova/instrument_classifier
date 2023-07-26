
## listen to one of the audio files in Jupyter:

#import IPython
#IPython.display.Audio("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/100bpm-808-like-drum-loop-74838.wav")



# ========================
### scipy.io.wavfile.read(somefile) returns a tuple of two items: the first is the sampling rate in samples per second, the second is a numpy array with all the data read from the file

import scipy
from scipy.io import wavfile
samplerate, data = wavfile.read("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/1.wav")



# =========================
# install librosa for conda: conda install -c conda-forge librosa
# (get librosa 0.10.0, for librosa < 0.8 get numba <= 0.49)

import librosa
import matplotlib.pyplot as plt


 # Load the audio as a waveform `y`
 # Store the sampling rate as `sr`

y, sr = librosa.load("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/1.wav")


# =============================
# import matplotlib properly =)

def create_fold_spectrograms (fold):
    spectrogram_path = '/Users/cookie/dev/instrumant_classifier/'
    audio_path = '/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/'
    print ('Processing fold {fold}')

    for audio_file in list (audio_path + '/' + fold + '.wav'):
        samples, sample_rate = librosa.load (audio_file, duration=5.0)
        fig = plt.figure(figsize=[0.72,0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible (False)
        ax. axes.get_yaxis().set_visible (False)
        ax.set_frame_on(False)
        filename = spectrogram_path/fold/Path(audio_file).name.replace('.wav',' .png')
        S = librosa. feature.melspectrogram (y=samples, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max))
        plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close('all')


# create_fold_spectrograms ('1')





# ==============================

# using scipy to draw spectrograms - sine wav example
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html

import numpy as np
from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
rng = np.random.default_rng()

fs = 10e3
N = 1e5
amp = 2 * np.sqrt(2)
noise_power = 0.01 * fs / 2
time = np.arange(N) / float(fs)
mod = 500*np.cos(2*np.pi*0.25*time)
carrier = amp * np.sin(2*np.pi*3e3*time + mod)
noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
noise *= np.exp(-time/5)
x = carrier + noise


f, t, Sxx = signal.spectrogram(x, fs)
plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()



# ==================================================
# Drawing a waveform

samplerate, data = wavfile.read("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/1.wav")

length = data.shape[0] / samplerate

import matplotlib.pyplot as plt
import numpy as np
time = np.linspace(0., length, data.shape[0])
plt.plot(time, data[:, 0], label="Left channel")
plt.plot(time, data[:, 1], label="Right channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()


# =================================================

# for this make sure that the file is mono, not stereo


# NB!!! smth is off with the spectrogram. check: normalize data?

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/1.wav")


mono_samples = samples.T[0]

frequencies, times, spectrogram = signal.spectrogram(mono_samples, sample_rate, return_onesided=False)

plt.pcolormesh(times, frequencies, spectrogram, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


# =============================================


# what's the minimum sample rate that would be good enough for classification? hete its 22050 Hz. it this too low?

import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio file
audio_file = '/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/1.wav'
y, sr = librosa.load(audio_file)

# Calculate and plot spectrogram
D = librosa.stft(y)
S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
plt.figure(figsize=(10, 5))
librosa.display.specshow(S_db, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()