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
import yml_processing

DICT_OF_LABELS = yml_processing.dict_of_labels()


def run_through_the_model(model_folder, X, header_text):

    import yaml
    model_address = model_folder + model_folder.split('/')[-2] + '.pt'
    models_yaml = model_folder + model_folder.split('/')[-2] + '.yml'
    with open(models_yaml, 'r') as file:
        yaml_input = yaml.safe_load(file)

    model = torch.load(model_address)
    model.eval()
    # get classes from yaml
    classes = utils.get_classes(models_yaml)


    # run through the model
    with torch.no_grad():
        X = X.to(device)
        output = model(X).cpu().numpy()

    print('Tensor output: ')
    print(output)

    txt_result = []
    txt_result.append(header_text)
    txt_result.append(np.array2string(output))
    txt_result.append('\n')

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

    txt_result.append('\n' + 'Percentages: ' + '\n' + for_popup + '\n')
    txt_result.append('==============================================')

    to_txt_file = '\n'.join(txt_result)

    return to_txt_file


def validation(folder_address, model_folder, trim = True):
    """ Takes in a folder of .wav files and a model. Runs the wavs from the folder through the models, saves the predictions into a yml and saves a png with a confusion matrix"""
    import yaml

    model_address = model_folder + model_folder.split('/')[-2] + '.pt'
    models_yaml = model_folder + model_folder.split('/')[-2] + '.yml'
    with open(models_yaml, 'r') as file:
        yaml_input = yaml.safe_load(file)
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
            if len(yaml_input['train_loop']['classes']) == 2:
                if file.split('#', 1)[0] == 'Full_Mix':
                        label = 'Full_Mix'
                else:
                        label = 'Solo'
            elif len(yaml_input['train_loop']['classes']) == 3:
                if file.split('#', 1)[0] == 'Full_Mix':
                    label = 'Full_Mix'
                elif file.split('#', 1)[0] == 'Drum_Kits':
                    label = 'Drum_Kits'
                else:
                    label == 'Solo'
            y_correct_label.append(label)
            y_correct_number.append(DICT_OF_LABELS[label])

            # run preprocessing
            samples, sr = audio_to_spectral_data.audio_to_samples(folder_address, file, sr=yaml_input['preprocessing']['sample_rate'])
            X = audio_to_spectral_data.audio_to_numpy(samples, models_yaml)

            a_mel_size = yaml_input['preprocessing']['mel_size']
            b_time_size = yaml_input['preprocessing']['time_size']


            X_empty = np.zeros(shape=(1, a_mel_size, b_time_size, 1), dtype=np.float32)
            X_empty[0] = X
            X = np.rollaxis(X_empty, 3, 1)
            X = torch.from_numpy(X.astype(np.float32))

            # run through the model
            with torch.no_grad():
                X = X.to(device)
                output = model(X).cpu().numpy()

            txt_header = '======================================================' + '\n'
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

    header_text = '==============================================' + '\n' + '\n'


    model_address = model_folder + model_folder.split('/')[-2] + '.pt'
    models_yaml = model_folder + model_folder.split('/')[-2] + '.yml'

    with open(models_yaml, 'r') as yml_file:
        yaml_input = yaml.safe_load(yml_file)
    a_mel_size = yaml_input['preprocessing']['mel_size']
    b_time_size = yaml_input['preprocessing']['time_size']


    # load model
    model = torch.load(model_address)
    model.eval()
    # get classes from yaml
    classes = utils.get_classes(models_yaml)

    torch.manual_seed(seed)
    if processing == False:
        if random == False:
            header_text += 'No preprocessing, input tensor = ' + str(input_num) + '\n'
            print(header_text)
            X = np.ones(shape=(1, 1, a_mel_size, b_time_size), dtype=np.float32) * input_num
            header_text += '\n' + 'sum: ' + str(X.sum()) + '\n' + 'min: ' + str(X.min()) + '\n' +'max: ' + str(X.max()) + '\n'

            plt.imshow(X[0,0,:,:], interpolation='none', origin='lower', aspect=32/(512 * 1.5))
            plt.savefig(model_folder + 'No_preprocessing_input_' + str(input_num), dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')

            X = torch.from_numpy(X.astype(np.float32))
        else:
            header_text += 'No preprocessing, input tensor RANDOM, seed = ' + str(seed) + '\n'
            print(header_text)
            #X = torch.randint(0, 80, (1, 1, 512, 32)).float()
            X = torch.rand(1, 1, a_mel_size, b_time_size)

            Y = X.numpy()

            plt.imshow(X[0,0,:,:], interpolation='none', origin='lower', aspect=32/(512 * 1.5))
            plt.savefig(model_folder + 'No_preprocessing_input_random_seed_' + str(seed), dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')

            header_text += '\n' + 'sum: ' + str(Y.sum()) + '\n' + 'min: ' + str(Y.min()) + '\n' +'max: ' + str(Y.max()) + '\n'

            print('\n')
    elif processing == True:
        if random == False:
            header_text += 'Preprocessed, input signal = ' + str(input_num) + '\n'
            print(header_text)
            # run preprocessing
            samples = np.ones(shape = (16000,)) * input_num
            X = audio_to_spectral_data.audio_to_numpy(samples, models_yaml)
            X_empty = np.zeros(shape=(1, 1, a_mel_size, b_time_size), dtype=np.float32)
            X_empty[0] = X

            X = np.rollaxis(X_empty, 3, 1)

            print('sum: ', X.sum())
            print('min: ', X.min())
            print('max: ', X.max())
            header_text += '\n' + 'sum: ' + str(X.sum()) + '\n' + 'min: ' + str(X.min()) + '\n' +'max: ' + str(X.max()) + '\n'

            plt.imshow(X[0,0,:,:], interpolation='none', origin='lower', aspect=32/(512 * 1.5))
            plt.savefig(model_folder + 'Preprocessed_input_' + str(input_num), dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')

            X = torch.from_numpy(X.astype(np.float32))

        else:
            header_text += 'Preprocessing, input signal RANDOM, seed = ' + str(seed) + '\n'
            print(header_text)
            # run preprocessing
            samples = torch.rand(16000,).numpy()
            X = audio_to_spectral_data.audio_to_numpy(samples, models_yaml)
            X_empty = np.zeros(shape=(1, 1, a_mel_size, b_time_size), dtype=np.float32)
            X_empty[0] = X
            X = np.rollaxis(X_empty, 3, 1)
            header_text += '\n' + 'sum: ' + str(X.sum()) + '\n' + 'min: ' + str(X.min()) + '\n' +'max: ' + str(X.max()) + '\n'

            plt.imshow(X[0,0,:,:], interpolation='none', origin='lower', aspect=32/(512 * 1.5))
            plt.savefig(model_folder + 'Preprocessed_input_random_seed' + str(seed), dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close('all')

            X = torch.from_numpy(X.astype(np.float32))

    elif processing > 0:

        header_text += 'Preprocessed, cut manually to 32 bands, input signal = ' + str(input_num) + ', number of seconds = ' + str(processing) + '\n'
        print(header_text)
        # run preprocessing
        samples = np.ones(shape = (32000,)) * input_num
        X = audio_to_spectral_data.audio_to_numpy(samples, models_yaml)
        X_empty = np.zeros(shape=(1, 1, a_mel_size, b_time_size), dtype=np.float32)

        for i in range(b):
            X_empty[0,:,i,:] = X[:,i,:]

        X = np.rollaxis(X_empty, 3, 1)

        header_text += '\n' + 'sum: ' + str(X.sum()) + '\n' + 'min: ' + str(X.min()) + '\n' +'max: ' + str(X.max()) + '\n'
        print('sum: ', X.sum())
        print('min: ', X.min())
        print('max: ', X.max())

        plt.imshow(X[0,0,:,:], interpolation='none', origin='lower', aspect=32/(512 * 1.5))
        plt.savefig(model_folder + 'Preprocessed_cut_manually_to_32_input_' + str(input_num), dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close('all')

        X = torch.from_numpy(X.astype(np.float32))


    to_txt_file = run_through_the_model(model_folder, X, header_text)

    return to_txt_file


def simple_validations(model_address):

    import yaml
    models_yaml = model_address + model_address.split('/')[-2] + '.yml'

    txt_result = ''

    # Zeroes
    txt_result += validation_with_numbers(model_address, input_num = 0, processing = False, random = False)
    txt_result += validation_with_numbers(model_address, input_num = 0, processing = True, random = False)
    # Ones
    txt_result += validation_with_numbers(model_address, input_num = 1, processing = False, random = False)
    txt_result += validation_with_numbers(model_address, input_num = 1, processing = True, random = False)
    # Random
    txt_result += validation_with_numbers(model_address, processing = False, random = True, seed = 0)
    txt_result += validation_with_numbers(model_address, processing = False, random = True, seed = 3)
    txt_result += validation_with_numbers(model_address, processing = True, random = True, seed = 0)
    txt_result += validation_with_numbers(model_address, processing = True, random = True, seed = 1)
    txt_result += validation_with_numbers(model_address, processing = True, random = True, seed = 2)
    txt_result += validation_with_numbers(model_address, processing = True, random = True, seed = 3)
    # Ones vs Cut ones
    txt_result += validation_with_numbers(model_address, input_num = 1, processing = 2, random = False)
    txt_result += validation_with_numbers(model_address, input_num = 1, processing = 1.1, random = False)
    txt_result += validation_with_numbers(model_address, input_num = 1, processing = 1.2, random = False)
    txt_result += validation_with_numbers(model_address, input_num = 1, processing = True, random = False)

    destination_address = models_yaml[:-4]
    with open(destination_address + '_simple_validations' + '.txt', 'w') as f:
        f.write(txt_result)




def translation_invariance(model_folder):

    # import parameters from yaml file
    model_address = model_folder + model_folder.split('/')[-2] + '.pt'
    import yaml
    models_yaml = model_folder + model_folder.split('/')[-2] + '.yml'
    with open(models_yaml, 'r') as yml_file:
        yaml_input = yaml.safe_load(yml_file)
    a_mel_size = yaml_input['preprocessing']['mel_size']
    b_time_size = yaml_input['preprocessing']['time_size']

    # load model
    model = torch.load(model_address)
    model.eval()
    # get classes from yaml
    classes = utils.get_classes(models_yaml)

    # 1. manufactured signal test without preprocessing



    header_text = 'No preprocessing' + '\n'
    print(header_text)

    for i in range(4):
        X = np.zeros(shape=(1, 1, a_mel_size, b_time_size), dtype=np.float32)
        X[0,0,:,i*7] = 1
        X[0,0,:,i*7+1] = 1
        X[0,0,:,i*7+2] = 1
        plt.imshow(X[0,0,:,:], interpolation='none', origin='lower', aspect=32/(512 * 1.5))
        plt.show()
        X_torch = torch.from_numpy(X.astype(np.float32))
        to_txt_file = run_through_the_model(model_folder, X_torch, header_text)


    # 2. real signal example, processed

    folder_address = '/Users/cookie/dev/instrument_classifier/unit_testing/for_validation/'



    for file in os.listdir(folder_address):
        if file[-4:].lower() == '.wav':

            # get label
            label = file.split('#', 1)[0]

            # import and convert to samples
            samples, sample_rate = audio_to_spectral_data.audio_to_samples(folder_address, file)
            # preprocess

            for i in range(2):

                # move a little bit
                multiplyer = int( b_time_size / 3  + 2)
                #pad_amount = int(sample_rate * seconds_multiplyer * i)
                pad_amount = 0
                padded_samples = np.pad(samples, pad_width=(pad_amount, pad_amount), mode = 'constant', constant_values=0)

                # preprocess
                array = audio_to_spectral_data.audio_to_numpy(padded_samples, models_yaml, librosa_export = False)

                # pad the spectrogram
                final_array = np.zeros(shape=(a_mel_size, b_time_size, 1))
                #final_array = array

                for time in range(b_time_size):

                    if time + i * multiplyer < b_time_size:
                        final_array[:, time + i * multiplyer] = array[:, time]

                filename = file + '_moved_' + str(i * multiplyer) + '_multiplyer'
                audio_to_spectral_data.numpy_to_mel_spectrogram(final_array,  filename, folder_address)

                # run model
                X = np.zeros(shape=(1, a_mel_size, b_time_size, 1),
                      dtype=np.float32)
                X[0] = final_array
                X = np.rollaxis(X, 3, 1)
                X_torch = torch.from_numpy(X.astype(np.float32))

                header_text = '\n' + 'Preprocessing, file: ' + filename
                print(header_text)
                run_through_the_model(model_folder, X_torch, header_text)

            print('==============================================')






'''
    ====================================================
                        MAIN
    ====================================================
'''



if __name__ == "__main__":

    '''
    folder_address = '/Users/cookie/dev/m_loudener/audio_files/_train_test_dataset/Validation_Set/'
    model_address = '/Users/cookie/dev/instrument_classifier/model_results/m_loudener/lr_0.0003_epochs_150_20240302-051316/'

    #validation(folder_address, model_address)

    folder_address_2 = '/Users/cookie/dev/m_loudener/audio_files/_train_test_dataset/validation_thingy/'

    #validation(folder_address_2, model_address, trim = False)

    simple_validations(model_address)

    '''

    translation_invariance('/Users/cookie/dev/instrument_classifier/model_results/m_loudener/lr_0.0003_epochs_220_20240323-064500/')

