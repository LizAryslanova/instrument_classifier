
## listen to one of the audio files in Jupyter:

import IPython
IPython.display.Audio("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/100bpm-808-like-drum-loop-74838.wav")



# ========================
### scipy.io.wavfile.read(somefile) returns a tuple of two items: the first is the sampling rate in samples per second, the second is a numpy array with all the data read from the file

#import scipy
# from scipy.io import wavfile
#samplerate, data = wavfile.read("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/100bpm-808-like-drum-loop-74838.wav")



# =========================
# install librosa for conda: conda install -c conda-forge librosa
# (get librosa 0.10.0, for librosa < 0.8 get numba <= 0.49)

import librosa


 # Load the audio as a waveform `y`
 # Store the sampling rate as `sr`

y, sr = librosa.load("/Users/cookie/dev/instrumant_classifier/audio_for_testing_import/100bpm-808-like-drum-loop-74838.wav")



