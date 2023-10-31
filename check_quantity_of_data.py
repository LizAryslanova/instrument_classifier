import numpy as np
import pickle
import utils
import os

current_dir = os.path.abspath(os.getcwd())
notes = '_no_split_test_'


with open(current_dir + '/pickles/kaggle_test_mel_8000' + notes + '_x', 'rb') as f:
    X_test = pickle.load(f)

with open(current_dir + '/pickles/kaggle_train_mel_8000' + notes + '_x', 'rb') as f:
    X_train = pickle.load(f)


test_size = np.shape(X_test)
train_size = np.shape(X_train)


print('Test = ', test_size)
print('Train = ', train_size)