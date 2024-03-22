
def audio_to_samples(folder_address, file, sr=16000):
    '''
        ===================================
        Takes in the *file* from the *folder_address*, checks if the file is a .wav and if True:
        Loads through librosa and returns samples and sample rate
        If sr = True it will use librosa's standart 22050 Hz, else it will use sr = None (and import with the file's sample rate). If sr is a positive number - that would be the forces sample rate.
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
        Return sample arrays that cuts the first seconds_to_cut
        ===================================
    '''
    cut_time = int(sample_rate * seconds_to_cut)
    cut_signal = samples[cut_time:]
    return cut_signal, sample_rate




def audio_to_numpy(samples, yaml_address, librosa_export = False):
    """ Converts samples (np.array) input into a (np.array) that is a mel spectrogram. Takes parameters from the yml file  """


    import librosa
    import librosa.display
    from librosa.effects import trim
    import numpy as np
    import yaml
    with open(yaml_address, 'r') as file:
        yaml_input = yaml.safe_load(file)

    # ===========================================
    # parameters
    sample_rate = yaml_input['preprocessing']['sample_rate']
    max_freq = yaml_input['preprocessing']['fmax']
    duration_s = yaml_input['preprocessing']['duration_sec']   # seconds_to_cut
    fft_size = yaml_input['preprocessing']['fft_size']
    fft_stride = yaml_input['preprocessing']['fft_hop']
    mel_bins = yaml_input['preprocessing']['mel_size']
    mel_power = yaml_input['preprocessing']['mel_power']
    num_samples = duration_s * sample_rate
    filter_dc = yaml_input['preprocessing']['filter_dc']
    top_db = yaml_input['preprocessing']['librosa_trim_db']
    stft_center = yaml_input['preprocessing']['stft_center']
    after_trim = yaml_input['preprocessing']['after_trim']   # loop or pad

    # ===========================================
    # trim the silence in the beginning (and end) of the file
    trimmed_signal, _ = librosa.effects.trim(samples, top_db=top_db)

    pad_amount = int ( fft_size / 2 )

    trimmed_padded_signal = np.pad(trimmed_signal, pad_width=(pad_amount, pad_amount), mode = 'edge')

    # cuts the duration_s seconds from the trimmed signal
    cut_time = int(sample_rate * duration_s)
    cut_signal = trimmed_padded_signal[0:cut_time]

    # normalizes data
    max_peak = np.max(np.abs(cut_signal))
    if max_peak != 0:
        ratio = 1 / max_peak
    else:
        ratio = 1
    normalised_signal = cut_signal * ratio

    # padding with 0 for things shorter that duration_s seconds
    if (len(normalised_signal) < cut_time):
        if after_trim == 'pad':
            normalised_signal = np.pad(normalised_signal, pad_width=(0, cut_time - len(normalised_signal)), mode = 'edge')
        elif after_trim == 'loop':
            print('DONT FORGET TO WRiTE THE LOOP PART: audio to spectral data, audio to numpy')
            # check how much to add ?
            # or just loop add the thing?


    # esporting the result after trim and normalise
    if librosa_export:
        import soundfile as sf
        sf.write('/Users/cookie/dev/m_loudener/AFTER_LIBROSA.wav', normalised_signal, sample_rate)

    # Use intermediate STFT to have the option of filtering DC out
    STFT_result = np.abs (librosa.stft (normalised_signal, n_fft=fft_size, hop_length=fft_stride, center = stft_center)) ** mel_power

    if filter_dc:
        STFT_result[0] = 0
        STFT_result[1] = 0 # first row contains the frequency of the windowing function

    mel_spec = librosa.feature.melspectrogram (S=STFT_result, sr=sample_rate, n_mels=mel_bins, fmax=max_freq)
    # Convert to decibels in [0, 80]
    mel_spec_db = librosa.amplitude_to_db (mel_spec, ref=1.0, top_db=80.0)
    mel_spec_db -= mel_spec_db.min()
    mel_sgram = mel_spec_db

    # creates a final_array. Can do greyslale data. For that change: channels = 3; multiplyer = 255
    a_mel_size, b_time_size = mel_sgram.shape

    # write time_size to yml

    yaml_input['preprocessing']['time_size'] = b_time_size

    with open(yaml_address, 'w') as yml_file:
        documents = yaml.dump(yaml_input, yml_file)

    channels = 1    # use 3 for RGB
    multiplyer = 1   # use 255 for greyscale RGB
    final_array = np.zeros(shape=(a_mel_size, b_time_size, channels), dtype=np.float32)
    for i in range(channels):
        final_array[:,:,i] = multiplyer * mel_sgram

    return final_array






def audio_to_mel_spectrogram(folder_address, file, yaml_address = '/Users/cookie/dev/instrument_classifier/model.yml'):
    """ Draws and saves mel spectrograms using audio_to_numpy calculations """


    import matplotlib.pyplot as plt
    import yaml
    import os

    with open(yaml_address, 'r') as yml_file:
        yaml_input = yaml.safe_load(yml_file)
    sample_rate = yaml_input['preprocessing']['sample_rate']
    destination_address = yaml_input['preprocessing']['address_for_mel_spectrograms']
    duration_sec = yaml_input['preprocessing']['duration_sec']
    fft_hop = yaml_input['preprocessing']['fft_hop']


    if (file[-4:].lower() == '.wav'):

        samples, _ = audio_to_samples(folder_address, file, sr = sample_rate)
        mel_spec_db = audio_to_numpy(samples, yaml_address)

        plt.imshow(mel_spec_db[:,:,0], interpolation='none', origin='lower', aspect=32/(512 * 1.5))

        # CREATE A FOLDER to save all mel spectrograms
        destination_folder = destination_address + str(duration_sec) + '_sec_' + str(fft_hop) + '_hop' + '/'

        if os.path.isdir(destination_folder) == False:
            os.mkdir(destination_folder)
            filename = destination_folder + file[:-4] + '_mel'+ '.png'

            plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')


        else:
            filename = destination_folder + file[:-4] + '_mel'+ '.png'
            plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')






def numpy_to_mel_spectrogram(numpy_array, file, address_to_save_spectrograms):

    import matplotlib.pyplot as plt
    import os

    plt.imshow(numpy_array[:,:,0], interpolation='none', origin='lower', aspect=32/(512 * 1.5))
    filename = address_to_save_spectrograms + file  + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')
