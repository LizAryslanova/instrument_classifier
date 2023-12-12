import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import os
import sys
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_dir)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
import utils
from cnn import CNN

folder_address = '/Users/cookie/dev/instrument_classifier/unit_testing/'
file1 = 'string_acoustic_075-090-100.wav'
file2 = 'G53-71607-1111-229.wav'


samples1, sample_rate1 = utils.audio_to_samples(folder_address, file1, sr = False)
#print(sample_rate1)

samples2, sample_rate2 = utils.audio_to_samples(folder_address, file2, sr = False)
#print(sample_rate2)


_, sample_rate_1a = utils.cut_audio_to_samples(samples1, sample_rate1, 1)
_, sample_rate_2a = utils.cut_audio_to_samples(samples2, sample_rate2, 1)


#print(sample_rate_1a)
#print(sample_rate_2a)



#utils.audio_to_spectrogram(folder_address, file1, folder_address, sr = False, seconds_to_cut=0.4)
#utils.audio_to_mel_spectrogram(folder_address, file1, folder_address, sr = False, seconds_to_cut=0.4)




# Shape of the spectrogram
def dim_of_spectrogram(file, sr = True, fmax = 8000):
    '''
        ==========================
        Returns dimensions of the spectrogram.
        Automatically takes sr as True - to use librosa's 22050 sample rate
        Use sr = False to take the file's sample rate.
        By default has fmax = 8000
        ==========================
    '''
    folder_address = current_dir + '/unit_testing/'
    file = 'G53-71607-1111-229.wav'
    samples, sample_rate = utils.audio_to_samples(folder_address, file, sr)
    fmax = 8000

    N = utils.audio_to_numpy(samples, sample_rate, fmax)
    return N.shape


print(dim_of_spectrogram(file1, sr=False, fmax=2000))
print(dim_of_spectrogram(file2, sr=False))