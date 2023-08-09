import os
import pandas as pd


kaggle_csv_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'


df = pd.read_csv(kaggle_csv_address)



print(pd.DataFrame(df, columns=['FileName', 'Class']))


print('======================')

'''
    Getting just the label from kaggle set .csv
'''

row_num = df[df['FileName'] == '2-E1-Minor 08' + '.wav'].index.to_numpy() # gets a row number of the entry (as a numpy array)
print(df['Class'][row_num[0]])  # gets the label from that row number


print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

file = '2-E1-Minor 08' + '.wav'


print ('Processing ' + file)
row_num = df[df['FileName'] == file[:-4] + '.wav'].index.to_numpy() # gets a row number of the entry (as a numpy array)

print(row_num)
print(row_num.shape)

print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


print('======================')





def get_label(file):

    kaggle_csv_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'
    df = pd.read_csv(kaggle_csv_address)


    row_num = df[df['FileName'] == file[:-4] + '.wav'].index.to_numpy() # gets a row number of the entry (as a numpy array)

    if row_num.shape == (1,):
        print ('Processing ' + file)
        label = df['Class'][row_num[0]]

    else:
        label = 'skip' # in case the fie is not in scv

    return label





kaggle_train_picture_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'
counter = 0

for file in os.listdir(kaggle_train_picture_address):

    if get_label(file) != 'skip':

        counter += 1
        print('Processing # = ' + str(counter) + '    ' + file)

