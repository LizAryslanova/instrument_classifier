import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

import utils
from cnn import CNN

from torch.optim import lr_scheduler


from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/')

import sys


'''
    ===================================
    Preparing the dataset
    ===================================
'''

classes = ('guitar', 'piano', 'drum', 'violin')

# Un-Pickle Test sets
with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_test_mel_x', 'rb') as f:
    X_test = pickle.load(f)
with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_test_mel_y', 'rb') as f:
    y_test = pickle.load(f)

X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test).type(torch.LongTensor)



class TrainSet(Dataset):

    def __init__(self, transform=None):
        # data loading
        with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_train_mel_x', 'rb') as f:
            X_train = pickle.load(f)
        with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_train_mel_y', 'rb') as f:
            y_train = pickle.load(f)

        self.x = torch.from_numpy(X_train.astype(np.float32))
        self.y = torch.from_numpy(y_train).type(torch.LongTensor)
        self.n_samples = X_train.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        #len(dataset)
        return self.n_samples

dataset = TrainSet()
train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)




'''
    ===================================
    Hyperparameters
    ===================================
'''

num_classes = 4
CNN_model = CNN()

learning_rate = 0.00001
num_epochs = 40

loss_function = nn.CrossEntropyLoss()  # softmax is included
optimizer = optim.SGD(CNN_model.parameters(), lr = learning_rate)

destination_address = '/Users/cookie/dev/instrumant_classifier/model_results/'



writer.add_graph(CNN_model, X_test)
# sys.exit()


'''
    ===================================
    Training loop
    ===================================
'''

n_total_steps = len(train_loader)
training_loss = []
test_loss = []
epoch_loss = []

running_loss = 0.0
running_correct = 0

scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):
        # forward pass and loss
        y_predicted = CNN_model(images)

        loss = loss_function(y_predicted, labels)

        optimizer.zero_grad()     # zero gradients
        loss.backward()
        optimizer.step()                 # updates

        running_loss += loss.item()
        epoch_loss.append(loss.item())
        '''
        outputs = CNN_model(X_test)
        _, predict = torch.max(outputs, 1)
        running_correct += (predict == labels).sum().item()
        '''

        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss/100, epoch * n_total_steps + i)
            #writer.add_scalar('accuracy', running_correct/100, epoch * n_total_steps + i)

            running_loss = 0.0
            running_correct = 0

    training_loss.append(sum(epoch_loss) / len(epoch_loss))
    epoch_loss = []


    # Calculating test loss
    with torch.no_grad():
        outputs = CNN_model(X_test)
        loss = loss_function(outputs, y_test)
        test_loss.append(loss.item())

    scheduler.step()


writer.close()

# Saving the Model
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
name = 'lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_' + timestr
filename = destination_address + name + '.pt'
torch.save(CNN_model, filename)


'''
    ===================================
    Testing
    ===================================
'''

import os
os.system('say "Cookie, I am plotting the picture." ')

y_predicted, accuracies, n_class_correct, n_class_samples = utils.test(CNN_model, X_test, y_test, classes)
utils.plot_image(training_loss, test_loss, num_epochs, learning_rate, classes, accuracies, y_test, y_predicted, filename, show = True)

