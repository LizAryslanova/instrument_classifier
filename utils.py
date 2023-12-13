
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import pickle


import os
current_dir = os.path.dirname(os.path.realpath(__file__))




def output_dimensions(width, height, padding, kernel, stride):
    '''
        ===================================
        Takes in dimensions of the input numpy array .shape[2] and .shape[3], padding, kernel and stride
            - calculates the dimentions of the output after a filter
            - returns hight and width output sizes
        ===================================
    '''
    output_width = int( ( width + 2 * padding - kernel ) / stride ) + 1
    output_height = int( ( height + 2 * padding - kernel ) / stride ) + 1

    return output_width, output_height



# Calculate dimensions for the nn.Linear layer
def dimensions_for_linear_layer(width, height):
    '''
        ===================================
        Takes in dimensions of the input numpy array (.shape[2] and .shape[3])
            - calculates the dimentions for the nn.Linear input
            - returns a product of hight and width output sizes
        ===================================
    '''

    return int(width * height)


# Calculate predicred values on a test set, calculate accuracies
def test(CNN_model, X_test, y_test, classes):
    '''
        ===================================
        Runs the test sets through the model without changing the gradients.
        Returns: predicted, accuracies (the first value - accuracy of the entire model, others are for individual instruments), n_class_correct, n_class_samples
        ===================================
    '''

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        num_classes = len(classes)
        n_class_correct = [0 for i in range(num_classes)]
        n_class_samples = [0 for i in range(num_classes)]


        outputs = CNN_model(X_test)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += y_test.size(0)
        n_correct += (predicted == y_test).sum().item()

        #print('Predicted = ', predicted)

        for i in range(X_test.shape[0]):   # number of files in the test set
            label = y_test[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        # print(f'Accuracy of the network: {acc} %')

        accuracies = []

        accuracies.append(acc)

        for i in range(num_classes):
            accuracies.append(100.0 * n_class_correct[i] / n_class_samples[i])
            # print(f'Accuracy of {classes[i]}: {acc} %')
            # print('n_class_correct = ', n_class_correct, ' n_class_samples = ', n_class_samples)

        return predicted, accuracies, n_class_correct, n_class_samples


# Creates a confusion matrix comparing test true labels with model predictions
def confusion_matrix(y_true, y_pred, num_classes):
    '''
        ===================================
        Takes true and predicted values for labels (as tensors)
        Returns numpy array with a confusion matrix whose i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
        ===================================
    '''
    from sklearn.metrics import confusion_matrix
    import sklearn

    labels = []

    for i in range(num_classes):
        labels.append(i)

    metric = sklearn.metrics.confusion_matrix(y_true.cpu(), y_pred.cpu(), labels=labels)
    return metric

# Plots a confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, width_scale, height_scale):
    '''
        ===================================
        Takes true and predicted values for labels (as tensors), tuple of classes
        plots (but doesn't show) the confusion matrix where Rows represent True labels and Columns represent - prebicted labels
        ===================================
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    num_classes = len(classes)

    confusion = confusion_matrix(y_true, y_pred, num_classes)

    column_headers = classes
    row_headers = classes

    cell_text = []
    for row in confusion:
        cell_text.append([f'{x}' for x in row])

    the_table = plt.table(cellText=cell_text,
                      rowLabels=row_headers,
                      rowLoc='right',
                      colLabels=column_headers, loc = 'center')

    the_table.scale(width_scale, height_scale)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)

    footer_text = '// Rows - true labels; Columns - predicted labels'
    plt.figtext(0.95, 0.05, footer_text, horizontalalignment='right', fontsize=8, weight='light')

    # Add title
    plt.title('Confusion Matrix')




def do_classification_report(y_true, y_pred, classes):
    import sklearn.metrics

    column_headers = ['Precision', 'Recall', 'f1 Score', 'Support']
    row_headers = list(classes) + ['Accuracy', 'Macro avg', 'Weighted avg']

    classification_report = sklearn.metrics.classification_report(y_true.cpu(), y_pred.cpu())
    lines = classification_report.split('\n')

    # removing empty lines from the lisi of lines
    for i in range(len(lines)):
        if len(lines) >= i:
            if len(lines[i]) == 0:
                lines.pop(i)

    # printing lines
    #for i in range(len(lines)):
    #    print (lines[i])

    #===================================
    # Creating table style (list) from the classification report (str)
    table_classes = lines[1:-3]
    table_summary = lines[-3:]
    table = []

    for line in table_classes:
        t = line.strip().split()
        table.append(t[1:])

    l1 = ['', ''] + table_summary[0].strip().split()[1:]
    table.append(l1)

    for i in range(2):
        table.append(table_summary[i+1].strip().split()[2:])

    #===================================

    return table, column_headers, row_headers



# Plots a confusion matrix
def plot_classification_report(y_true, y_pred, classes, width_scale, height_scale):
    '''
        ===================================
        Takes true and predicted values for labels (as tensors), tuple of classes
        plots (but doesn't show) the confusion matrix where Rows represent True labels and Columns represent - prebicted labels
        ===================================
    '''
    import numpy as np
    import matplotlib.pyplot as plt

    num_classes = len(classes)
    confusion_table, column_headers, row_headers = do_classification_report(y_true, y_pred, classes)

    cell_text = []

    for row in confusion_table:
        cell_text.append([f'{x}' for x in row])

    the_table = plt.table(cellText=cell_text, rowLabels=row_headers,
                      rowLoc='right', colLabels=column_headers, loc = 'center')

    the_table.scale(width_scale, height_scale)
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)

    # Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    # Hide axes border
    plt.box(on=None)

    # Add title
    plt.title('Classification Report')


# Plots and saves the final image with losses on epochs, accuracies and a confusion matrix
def plot_image(training_loss, test_loss, num_epochs, learning_rate, classes, accuracies, y_true, y_predicted, filename, show = False):
    '''
        ===================================
        Plots accuracies for all epochs, table with final accuracies of the model and individual instruments, a confusion matrix where Rows represent True labels and Columns represent - prebicted labels.
        Imports all the necessary things.

        Takes as input:
            - train losses (list of floats)
            - test losses (list of floats)
            - number of epochs (int)
            - learning rate (float)
            - all classes (tuple)
            - all accuracies (list)
        ===================================
    '''
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np

    #===============================

    accuracy = ('Accuracy', )
    row_Labels = accuracy + classes

    acc = []
    for item in accuracies:
        acc.append(str(round(item,1)) + ' %')

    columns = 1
    y_offset = np.zeros(columns)
    cell_text = []
    for row in range(len(acc)):
        y_offset = [acc[row]]
        cell_text.append([x for x in y_offset])

    #===============================

    fig = plt.figure(figsize=(18, 6), layout="constrained")
    fig.suptitle(f'Loss functions for {num_epochs} epochs, learning rate = {str(learning_rate)}', fontsize = 24)

    #===============================

    gs = GridSpec(nrows=2, ncols=4)

    #===============================
    # Graph

    ax_graphs = fig.add_subplot(gs[:, 0:2])
    ax_graphs.plot(training_loss, 'g', label='Training Loss')
    ax_graphs.plot(test_loss, 'r', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    #===============================
    # Table with accuracies

    ax_table = fig.add_subplot(gs[0, -2])
    ax_table.set_axis_off()
    the_table = plt.table(cellText=cell_text,
                          colWidths = [0.5] * 1,
                          rowLabels=row_Labels, loc = 'center')

    for (row, col), cell in the_table.get_celld().items():
        if (row == 0):
            cell.set_text_props(weight='bold')

    width_scale_acc, height_scale_acc = 0.8, 2
    the_table.scale(width_scale_acc, height_scale_acc)
    the_table.set_fontsize(12)

    #===============================
    # Confunsion matrix
    width_scale_conf, height_scale_conf = 1.5, 1.7
    ax_table2 = fig.add_subplot(gs[1, -1])
    the_table_2 = plot_confusion_matrix(y_true, y_predicted, classes, width_scale_conf, height_scale_conf)


    #===============================
    # Classification report

    width_scale_reprt, height_scale_reprt = 1.2, 1.3
    ax_table3 = fig.add_subplot(gs[0, -1])
    the_table3 = plot_classification_report(y_true, y_predicted, classes, width_scale_reprt, height_scale_reprt)


    #===============================
    # Saving and naming the image

    filename = filename[:-3]  + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0.1)

    if show == True:
        plt.show()
    plt.close('all')




def get_labels_from_nsynth():
    '''
        ===================================
        Returns a list of all labels from nsynth sataset
        ===================================
    '''
    import os
    nsynth_labels = []
    big_folder = current_dir + '/audio_files/nsynth/Training_audios/'

    for i in range(31):
        folder_address = big_folder + str(i+1) + '/'
        for file in os.listdir(folder_address):
            if file[-4:] == '.wav':
                label = file[:-16]
                if label not in nsynth_labels:
                    nsynth_labels.append(label)
    return nsynth_labels





def audio_to_samples(folder_address, file, sr=True):
    '''
        ===================================
        Takes in the *file* from the *folder_address*, checks if the file is a .wav and if True:
        Loads through librosa and returns samples and sample rate
        If sr = True it will use librosa's standart 22050 Hz, else it will use sr = None (and import with the file's sample rate)
        ===================================
    '''
    import librosa
    import librosa.display
    from librosa.effects import trim
    import numpy as np

    if (file[-4:] == '.wav'):
        audio_file = folder_address + file
        if sr:
            samples, sample_rate = librosa.load(audio_file)
        else:
            samples, sample_rate = librosa.load(audio_file, sr=None)

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

    sr = sample_rate    # sample rate, used to cut 3 seconds. 22050 for auto librosa choice
    cut_time = int(sr * seconds_to_cut)
    cut_signal = trimmed_signal[0:cut_time]

    # normalize data
    max_peak = np.max(np.abs(cut_signal))
    ratio = 1 / max_peak
    normalised_signal = cut_signal * ratio

    # padding with 0 for things shorter that 3 seconds
    if (len(normalised_signal) < cut_time):
        normalised_signal = np.pad(normalised_signal, pad_width=(0, cut_time - len(normalised_signal)))

    STFT_result = librosa.stft(normalised_signal)
    STFT_abs = np.abs(STFT_result)


    # MEL Spectrogram
    sgram_mag, _ = librosa.magphase(STFT_result)
    mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=sample_rate, n_mels=512, fmax = fmax)
    mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)

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




# Shape of the spectrogram
def dim_of_spectrogram():
    '''
        ==========================
        Returns dimensions of the spectrogram.
        ==========================
    '''
    folder_address = current_dir + '/unit_testing/'
    file = 'G53-71607-1111-229.wav'
    samples, sample_rate = audio_to_samples(folder_address, file)
    fmax = 8000

    N = audio_to_numpy(samples, sample_rate, fmax)
    return N.shape


def get_classes(address = 'model.yml'):
    import yaml
    with open(address, 'r') as file:
        yaml_input = yaml.safe_load(file)

    classes = ()
    for i in range(yaml_input['model_parameters']['num_classes']):
        classes = classes + (yaml_input['train_loop']['classes']['c_'+str(i+1)], )
    return classes

