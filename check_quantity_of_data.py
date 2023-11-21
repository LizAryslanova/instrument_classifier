import numpy as np
import pickle
import utils
import os
import yaml

current_dir = os.path.dirname(os.path.realpath(__file__))

with open('model.yml', 'r') as file:
    yaml_input = yaml.safe_load(file)



with open(current_dir + yaml_input['train_loop']['x_test_address'], 'rb') as f:
    X_test = pickle.load(f)
with open(current_dir + yaml_input['train_loop']['x_train_address'], 'rb') as f:
    X_train = pickle.load(f)



test_size = np.shape(X_test)
train_size = np.shape(X_train)


print('Test = ', test_size)
print('Train = ', train_size)