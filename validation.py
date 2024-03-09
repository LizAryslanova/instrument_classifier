from pathlib import Path
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
#device = torch.device('cpu')
import plotting_results

DICT_OF_LABELS = {

    'Drum_Kits': 0,
    'Drums_Kick': 1,
    'Drums_Tom': 2,
    'Drums_Snare': 3,
    'Drums_Hat': 4,

    'Vocals_Female': 5,
    'Vocals_Male': 6,

    'Bass': 7,
    'Guitar_Acoustic': 8,
    'Guitar_Electric': 9,
    'Piano': 10,
    'String': 11,

    'Brass': 12,
    'Flute': 13
}

def validation(folder_address, model_folder, trim = True):
    import yaml

    model_address = model_folder + model_folder.split('/')[-2] + '.pt'
    models_yaml = model_folder + model_folder.split('/')[-2] + '.yml'
    # load model
    model = torch.load(model_address)
    model.eval()
    # get classes from yaml
    classes = utils.get_classes(models_yaml)

    list_of_results = []
    txt_result = []
    y_predicted_number = []
    y_correct_label = []
    y_predicted_label = []
    y_correct_number = []

    # open folder
    for file in os.listdir(folder_address):
        if file[-4:].lower() == '.wav':

            # get the label
            label = file.split('#', 1)[0]
            y_correct_label.append(label)
            y_correct_number.append(DICT_OF_LABELS[label])

            # run preprocessing
            samples, sr = audio_to_spectral_data.audio_to_samples('', folder_address + file, sr=16000)
            X = audio_to_spectral_data.audio_to_numpy(samples, sr, 8000, seconds_to_cut=1, trim = trim)

            a, b, c = X.shape
            X_empty = np.zeros(shape=(1, a, b, c), dtype=np.float32)
            X_empty[0] = X
            X = np.rollaxis(X_empty, 3, 1)
            X = torch.from_numpy(X.astype(np.float32))

            # run through the model
            with torch.no_grad():
                X = X.to(device)
                output = model(X).cpu().numpy()

            txt_header = ' ======================================================' + '\n'
            txt_header += ' File name = ' + str(file) + '\n'
            txt_header += '-----------------------' + '\n'
            txt_result.append(txt_header)
            txt_result.append(np.array2string(output))
            txt_result.append('\n')

            # modify results to show %
            output[output < 0] = 0    # replaces negatives with 0
            sum_of_output = np.sum(output)
            output = np.round(100 * output / sum_of_output, decimals = 2)

            output_ascending_indices = output.argsort()
            output_descending_indices = np.flip(output_ascending_indices)

            dict_result = {}
            dict_result[' ================='] = '=================='
            dict_result[' File name'] = file
            dict_result['=================='] = '=================='
            dict_result[' File name'] = file
            dict_result[' True label '] = label

            for_popup = ''
            for i in range(len(output[0])):
                index = output_descending_indices[0][i]
                if output[0][index] != 0:
                    for_popup += str(classes[index]) + ' = ' + str(output[0][index]) + '%' + '\n'
                    #index = output_descending_indices[0][i]
                    #dict_result[str(classes[index])] = str(output[0][index])

            y_predicted_number.append(output_descending_indices[0][0])
            y_predicted_label.append(classes[output_descending_indices[0][0]])
            list_of_results.append(dict_result)
            list_of_results.append(for_popup)


    yaml_input = list_of_results
    destination_address = models_yaml[:-4]
    name = '_validation'
    with open(destination_address + name + ".yml", 'w') as file:
        documents = yaml.dump(yaml_input, file)
    to_txt_file = '\n'.join(txt_result)
    with open(destination_address + name + '_tensor' + '.txt', 'w') as f:
        f.write(to_txt_file)

    #Plotting confusion matrix
    y_correct_number_tensor = torch.Tensor(y_correct_number)
    y_predicted_number_tensor = torch.Tensor(y_predicted_number)
    the_table = plotting_results.plot_confusion_matrix(y_correct_number_tensor, y_predicted_number_tensor, classes, width_scale=1.5, height_scale=1)
    # Saving and naming the image
    filename = destination_address + name + '_validation_confusion_matrix' + '.png'
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0.1)




def validation_with_numbers(model_folder, input_num = 1, processing = False, random = False, seed = 0):
    import yaml

    print('==============================================' + '\n')

    model_address = model_folder + model_folder.split('/')[-2] + '.pt'
    models_yaml = model_folder + model_folder.split('/')[-2] + '.yml'
    # load model
    model = torch.load(model_address)
    model.eval()
    # get classes from yaml
    classes = utils.get_classes(models_yaml)

    torch.manual_seed(seed)
    if processing == False:
        if random == False:
            print('No preprocessing, input tensor =', input_num, '\n')
            X = np.ones(shape=(1, 1, 512, 32), dtype=np.float32) * input_num
            X = torch.from_numpy(X.astype(np.float32))
        else:
            print('No preprocessing, input tensor RANDOM, seed =', seed, '\n')
            #X = torch.randint(0, 80, (1, 1, 512, 32)).float()
            X = torch.rand(1, 1, 512, 32)
            print('sum: ', X.sum())
            print('min: ', X.min())
            print('max: ', X.max())
            print('\n')
    else:
        if random == False:
            print('Preprocessed, input signal =', input_num, '\n')
            # run preprocessing
            samples = np.ones(shape = (72000,)) * input_num
            sr = 16000
            X = audio_to_spectral_data.audio_to_numpy(samples, sr, 8000, seconds_to_cut=1, trim = trim)
            a, b, c = X.shape
            X_empty = np.zeros(shape=(1, a, b, c), dtype=np.float32)
            X_empty[0] = X
            X = np.rollaxis(X_empty, 3, 1)
            X = torch.from_numpy(X.astype(np.float32))

        else:
            print('Preprocessing, input signal RANDOM, seed =', seed, '\n')
            # run preprocessing
            samples = torch.rand(72000,).numpy()
            sr = 16000
            X = audio_to_spectral_data.audio_to_numpy(samples, sr, 8000, seconds_to_cut=1, trim = trim)
            a, b, c = X.shape
            X_empty = np.zeros(shape=(1, a, b, c), dtype=np.float32)
            X_empty[0] = X
            X = np.rollaxis(X_empty, 3, 1)
            X = torch.from_numpy(X.astype(np.float32))


    # run through the model
    with torch.no_grad():
        X = X.to(device)
        output = model(X).cpu().numpy()

    print('Tensor output: ')
    print(output)

    # modify results to show %
    output[output < 0] = 0    # replaces negatives with 0
    sum_of_output = np.sum(output)
    output = np.round(100 * output / sum_of_output, decimals = 2)

    output_ascending_indices = output.argsort()
    output_descending_indices = np.flip(output_ascending_indices)


    for_popup = ''

    for i in range(len(output[0])):
        index = output_descending_indices[0][i]
        if output[0][index] != 0:
            for_popup += str(classes[index]) + ' = ' + str(output[0][index]) + '%' + '\n'
            #index = output_descending_indices[0][i]
            #dict_result[str(classes[index])] = str(output[0][index])

    print('\n' + 'Percentages: ' + '\n' + for_popup + '\n')
    print('==============================================')






'''
    ====================================================
                        MAIN
    ====================================================
'''



if __name__ == "__main__":


    folder_address = '/Users/cookie/dev/m_loudener/audio_files/_train_test_dataset/Validation_Set/'
    model_address = '/Users/cookie/dev/instrument_classifier/model_results/m_loudener/lr_0.0003_epochs_150_20240302-051316/'

    #validation(folder_address, model_address)

    folder_address_2 = '/Users/cookie/dev/m_loudener/audio_files/_train_test_dataset/validation_thingy/'

    #validation(folder_address_2, model_address, trim = False)




    # Zeroes
    validation_with_numbers(model_address, input_num = 0, processing = False, random = False)
    validation_with_numbers(model_address, input_num = 0, processing = True, random = False)

    # Ones
    validation_with_numbers(model_address, input_num = 1, processing = False, random = False)
    validation_with_numbers(model_address, input_num = 1, processing = True, random = False)

    # Random
    validation_with_numbers(model_address, processing = False, random = True, seed = 0)
    validation_with_numbers(model_address, processing = False, random = True, seed = 3)



