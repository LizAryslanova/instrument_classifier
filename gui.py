from pathlib import Path
import PySimpleGUI as sg
import os
import matplotlib.pyplot as plt     # IMPORTANT! Import matplotlib before librosa !!!
import librosa
import librosa.display
from librosa.effects import trim
import numpy as np

import utils
import audio_to_spectral_data

import torch
import torchvision
import pickle

from cnn import CNN

import sys
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def popup_text(filename, text):

    layout = [
        [sg.Multiline(text, size=(80, 25)),],
    ]
    win = sg.Window(filename, layout, modal=True, finalize=True)

    while True:
        event, values = win.read()
        if event == sg.WINDOW_CLOSED:
            break
    win.close()

sg.theme("DarkBlue3")
sg.set_options(font=("Microsoft JhengHei", 16))

layout = [
    [
        sg.Input(key='-INPUT-'),
        sg.FileBrowse(file_types=(("WAV Files", "*.wav"), ("ALL Files", "*.*"))),
        sg.Button("Process"),
    ]
]

window = sg.Window('Title', layout)

while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break
    elif event == 'Process':
        filename = values['-INPUT-']
        if Path(filename).is_file():
            try:
                with open(filename, "rt", encoding='utf-8') as f:

                    samples, sr = audio_to_spectral_data.audio_to_samples('', filename, sr=False)
                    X = audio_to_spectral_data.audio_to_numpy(samples, 16000, 8000, seconds_to_cut=1)

                    current_dir = os.path.dirname(os.path.realpath(__file__))
                    # model built for 1 second
                    model = torch.load(current_dir + '/model_results/lr_0.0002_epochs_100_20240202-185037.pt')

                    a, b, c = X.shape
                    X_empty = np.zeros(shape=(1, a, b, c),
                     dtype=np.float32)

                    X_empty[0] = X
                    X = np.rollaxis(X_empty, 3, 1)
                    X = torch.from_numpy(X.astype(np.float32))

                    with torch.no_grad():
                        X = X.to(device)
                        output = model(X).cpu().numpy()

                    #classes = utils.get_classes()
                    classes = ('Electronic Bass', 'Acoustic Strings', 'Acoustic Mallet', 'Acoustic Keyboard', 'Acoustic Guitar', 'Vocals')

                    output[output < 0] = 0    # replaces negatives with 0
                    sum_of_output = np.sum(output)
                    output = np.round(100 * output / sum_of_output, decimals = 2)

                    output_ascending_indices = output.argsort()
                    output_descending_indices = np.flip(output_ascending_indices)

                    for_popup = ''

                    for i in range(len(output[0])):
                        index = output_descending_indices[0][i]
                        for_popup += str(classes[index]) + ' = ' + str(output[0][index]) + '%' + '\n'

                    #for_popup = str(output) + '\n' + str(classes)


                    filename_for_popup = filename.split('/')
                    print('filename = ', filename)


                popup_text(filename_for_popup[-1], for_popup)
            except Exception as e:
                print("Error: ", e)

window.close()
