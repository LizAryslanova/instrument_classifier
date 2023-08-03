# ========================
### scipy.io.wavfile.read(somefile) returns a tuple of two items: the first is the sampling rate in samples per second, the second is a numpy array with all the data read from the file

import scipy
from scipy.io import wavfile
samplerate, data = wavfile.read("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/1.wav")



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






# =================================================

# for this make sure that the file is mono, not stereo


# NB!!! smth is off with the spectrogram. check: normalize data?

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

sample_rate, samples = wavfile.read("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/1.wav")


mono_samples = samples[:, 0]

frequencies, times, spectrogram = signal.spectrogram(mono_samples, sample_rate)

plt.pcolormesh(times, frequencies[0:20], spectrogram[0:20], shading='gouraud')
plt.title('Spectrogram - Scipy')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


# =============================================