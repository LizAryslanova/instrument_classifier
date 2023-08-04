import pandas as pd


kaggle_csv_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'


df = pd.read_csv(kaggle_csv_address)



print(pd.DataFrame(df, columns=['FileName', 'Class']))


print(df.shape)

kaggle_train_picture_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'


result = pd.DataFrame(df[ df['FileName'] == '2-E1-Minor 08' + '.wav' ], columns=['Class'])

print('this one: ', result)

# reading all files from foder and asigning them labels from csv

def load_data(address):

    for file in os.listdir(address):
        print ('Processing ' + file)

        result = df[ df['FileName'] == file[:-4] + '.wav' ]
