import torch
import torchvision
import os
import pickle
import numpy as np

from cnn import CNN

current_dir = os.path.dirname(os.path.realpath(__file__))
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# model built for .5 seconds
model = torch.load(current_dir + '/model_results/lr_0.0002_epochs_140_20231219-132849.pt')





# Un-Pickle Test sets
with open(current_dir + '/pickles/nsynth_test_x_new_sr_05sec', 'rb') as f:
    X_test = pickle.load(f)

_, a, b, c = X_test.shape

X_test = torch.from_numpy(X_test.astype(np.float32)).to(device)

print(X_test.shape)

example = X_test

print(example.shape)

# example = torch.rand(1, a, b, c).to(device)



traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("name.pt")
