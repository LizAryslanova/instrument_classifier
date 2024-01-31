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
import os
import pickle
import numpy as np
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
                    X = audio_to_spectral_data.audio_to_numpy(samples, sr, 8000, seconds_to_cut=0.5)

                    current_dir = os.path.dirname(os.path.realpath(__file__))
                    # model built for .5 seconds
                    model = torch.load(current_dir + '/model_results/lr_0.0002_epochs_140_20231219-132849.pt')


                    a, b, c = X.shape
                    X_empty = np.zeros(shape=(1, a, b, c),
                     dtype=np.float32)

                    print(X_empty.shape)

                    X_empty[0] = X

                    X = np.rollaxis(X_empty, 3, 1)

                    X = torch.from_numpy(X.astype(np.float32))



                    with torch.no_grad():
                        X = X.to(device)
                        output = model(X)


                    print(X.shape)




                popup_text(filename, output)
            except Exception as e:
                print("Error: ", e)

window.close()
