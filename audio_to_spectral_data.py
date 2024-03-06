'''
Contains functions:
    audio_to_samples(folder_address, file, sr=True)
    cut_audio_to_samples(samples, sample_rate, seconds_to_cut)
    audio_to_numpy(samples, sample_rate, fmax, seconds_to_cut = 3)
    audio_to_mel_spectrogram(folder_address, file, destination_address, sr = True, seconds_to_cut = 3)
    audio_to_spectrogram(folder_address, file, destination_address, sr = True, seconds_to_cut = 3)
'''



def audio_to_samples(folder_address, file, sr=True):
    '''
        ===================================
        Takes in the *file* from the *folder_address*, checks if the file is a .wav and if True:
        Loads through librosa and returns samples and sample rate
        If sr = True it will use librosa's standart 22050 Hz, else it will use sr = None (and import with the file's sample rate)
        If the file is recorded in stereo - at averages between channels and returns mono
        ===================================
    '''
    import librosa
    import librosa.display
    from librosa.effects import trim
    import numpy as np

    if (file[-4:].lower() == '.wav'):
        audio_file = folder_address + file
        if sr == True:
            samples, sample_rate = librosa.load(audio_file, mono = False)
        elif sr == False:
            samples, sample_rate = librosa.load(audio_file, sr=None, mono = False)
        elif sr > 0:
            samples, sample_rate = librosa.load(audio_file, sr=sr, mono = False)


        if samples.shape[0] == 2:
            averaged_samples = ( samples[0] + samples[1] ) / 2
            samples = averaged_samples

    return samples, sample_rate



def cut_audio_to_samples(samples, sample_rate, seconds_to_cut):
    '''
        ===================================
        Takes in the samples and sample rate (from audio_to_samples function)
        Return sampkle arrays that cuts the first seconds_to_cut
        ===================================
    '''

    cut_time = int(sample_rate * seconds_to_cut)
    cut_signal = samples[cut_time:]

    return cut_signal, sample_rate




def audio_to_numpy(samples, sample_rate, fmax, seconds_to_cut = 3):

    '''
        ===================================
        Takes in the samples and sample rate (from audio_to_samples function)
            - trims the silence,
            - takes the first 3 seconds
            - padds for files shorter than 3 seconds
            - normalises
            - stft
            - returns a numpy array of absolute values after stft
        ===================================
    '''

    import librosa
    import librosa.display
    from librosa.effects import trim
    import numpy as np


    trimmed_signal, _ = librosa.effects.trim(samples, top_db=15)

    # sample rate, used to cut 3 seconds. 22050 for auto librosa choice
    cut_time = int(sample_rate * seconds_to_cut)
    cut_signal = trimmed_signal[0:cut_time]

    # normalize data
    max_peak = np.max(np.abs(cut_signal))
    ratio = 1 / max_peak
    normalised_signal = cut_signal * ratio

    # padding with 0 for things shorter that 3 seconds
    if (len(normalised_signal) < cut_time):
        normalised_signal = np.pad(normalised_signal, pad_width=(0, cut_time - len(normalised_signal)))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Parameters
    sample_rate = 16_000
    max_freq = 8_000
    duration_s = 1
    fft_size = 2048
    fft_stride = 512
    mel_bins = 512
    num_samples = duration_s * sample_rate

    # This provides the exact same result as the code currently used, I recommend switching to this
    mel_spec = librosa.feature.melspectrogram (y=normalised_signal, sr=sample_rate, n_fft=fft_size, hop_length=fft_stride, n_mels=mel_bins, power=1.0, fmax=max_freq)

    # Convert to decibels in [0, 80]
    mel_spec_db = librosa.amplitude_to_db (mel_spec, ref=1.0, top_db=80.0)
    mel_spec_db -= mel_spec_db.min()
    mel_sgram = mel_spec_db


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    '''
    STFT_result = librosa.stft(normalised_signal)
    STFT_abs = np.abs(STFT_result)
    # MEL Spectrogram
    sgram_mag, _ = librosa.magphase(STFT_result)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate, n_mels=512, fmax = fmax)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
    '''
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #print(STFT_abs.shape)
    #print(STFT_abs[10,50])

    #a, b = STFT_result.shape
    a, b = mel_sgram.shape
    channels = 1    # use 3 for RGB
    multiplyer = 1   # use 255 for greyscale RGB


    final_array = np.zeros(shape=(a, b, channels), dtype=np.float32)

    for i in range(channels):
        # img[:,:,i] = multiplyer * D_abs
        # final_array[:,:,i] = multiplyer * STFT_abs
        final_array[:,:,i] = multiplyer * mel_sgram

    return final_array








def audio_to_mel_spectrogram(folder_address, file, destination_address, sr = True, seconds_to_cut = 3):
    '''
        ===================================
        If sr = True it will use librosa's standart 22050 Hz, else it will use sr = None (and import with the file's sample rate)

        Takes in the *file* from the *folder_address*, checks if the file is a .wav and if True:
            - trims the silence,
            - takes the first 3 seconds
            - adds 0 padding if the file is shorter
            - creates a spectrogram image
            - saves image destination_address with the same name as the original file (.png)
        ===================================
    '''
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    from librosa.effects import trim
    import numpy as np

    if (file[-4:] == '.wav'):

        audio_file = folder_address + file

        if sr:
            samples, sample_rate = librosa.load(audio_file)
        else:
            samples, sample_rate = librosa.load(audio_file, sr = None)
        trimed_signal, _ = librosa.effects.trim(samples, top_db=15)


        sr = sample_rate # sample rate, used to cut 3 seconds
        cut_time = int(sr * seconds_to_cut)
        cut_signal = trimed_signal[0:cut_time]

        # normalize data
        max_peak = np.max(np.abs(cut_signal))
        ratio = 1 / max_peak
        normalised_signal = cut_signal * ratio

        # padding with 0 for things shorter that 3 seconds
        if (len(normalised_signal) < cut_time):
            cut_signal = np.pad(normalised_signal, pad_width=(0, cut_time - len(cut_signal)))


        fig = plt.figure(figsize=[1, 1])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible (False)
        ax.axes.get_yaxis().set_visible (False)
        ax.set_frame_on(False)
        filename = destination_address + file[:-4] + '_mel'+ '.png'

        STFT_result = librosa.stft(cut_signal, n_fft = 2048)

        # MEL Spectrogram
        sgram_mag, _ = librosa.magphase(STFT_result)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate, n_mels=512, fmax = 8000)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

        librosa.display.specshow(mel_sgram, x_axis='time', y_axis='off', sr = sample_rate)

        plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close('all')



def audio_to_spectrogram(folder_address, file, destination_address, sr = True, seconds_to_cut = 3):
    '''
        ===================================
        Takes in the *file* from the *folder_address*, checks if the file is a .wav and if True:
            - trims the silence,
            - takes the first 3 seconds
            - adds 0 padding if the file is shorter
            - creates a spectrogram image
            - saves image destination_address with the same name as the original file (.png)
        ===================================
    '''
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    from librosa.effects import trim
    import numpy as np

    if (file[-4:] == '.wav'):

        audio_file = folder_address + file

        if sr:
            samples, sample_rate = librosa.load(audio_file)
        else:
            samples, sample_rate = librosa.load(audio_file, sr = None)

        samples, sample_rate = librosa.load(audio_file)
        trimed_signal, _ = librosa.effects.trim(samples, top_db=15)


        sr = sample_rate # sample rate, used to cut 3 seconds

        cut_time = int(sr * seconds_to_cut)
        cut_signal = trimed_signal[0:cut_time]


        # normalize data
        max_peak = np.max(np.abs(cut_signal))
        ratio = 1 / max_peak
        normalised_signal = cut_signal * ratio


        # padding with 0 for things shorter that 3 seconds
        if (len(normalised_signal) < cut_time):
            cut_signal = np.pad(normalised_signal, pad_width=(0, cut_time - len(cut_signal)))


        fig = plt.figure(figsize=[1, 1])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible (False)
        ax.axes.get_yaxis().set_visible (False)
        ax.set_frame_on(False)
        filename = destination_address + file[:-4] + '.png'

        D = librosa.stft(cut_signal)
        S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
        librosa.display.specshow(S_db, x_axis='time', y_axis='off')

        plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close('all')


