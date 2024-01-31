import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

import utils
import plotting_results
import data_management

from cnn import CNN
from torch.optim import lr_scheduler
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/')
import sys
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
import yaml
with open('model.yml', 'r') as file:
    yaml_input = yaml.safe_load(file)

import sklearn


'''
    ===================================
    Preparing the dataset
    ===================================
'''

classes = utils.get_classes()



# Un-Pickle Test sets
with open(current_dir + yaml_input['train_loop']['x_test_address'], 'rb') as f:
    X_test = pickle.load(f)
with open(current_dir + yaml_input['train_loop']['y_test_address'], 'rb') as f:
    y_test = pickle.load(f)

X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test).type(torch.LongTensor)


# data loading
with open(current_dir + yaml_input['train_loop']['x_train_address'], 'rb') as f:
    X_train = pickle.load(f)
with open(current_dir + yaml_input['train_loop']['y_train_address'], 'rb') as f:
    y_train = pickle.load(f)


dataset = data_management.TrainSet(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=4, shuffle=True, num_workers=0)




'''
    ===================================
    Hyperparameters
    ===================================
'''

image_width = X_test.shape[2]
image_height = X_test.shape[3]

num_classes = yaml_input['model_parameters']['num_classes']
CNN_model = CNN(image_width, image_height, **yaml_input['model_parameters']).to(device)
learning_rate = yaml_input['train_loop']['learning_rate']
num_epochs = yaml_input['train_loop']['num_epochs']
destination_address = current_dir + yaml_input['train_loop']['model_results_address']

loss_function = nn.CrossEntropyLoss()  # softmax is included
optimizer = optim.SGD(CNN_model.parameters(), lr = learning_rate)





# writer.add_graph(CNN_model, X_test)
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

scheduler = lr_scheduler.StepLR(optimizer, step_size=yaml_input['train_loop']['step_size'], gamma=yaml_input['train_loop']['gamma'])

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # forward pass and loss

        y_predicted = CNN_model(images)

        loss = loss_function(y_predicted, labels)

        # Regularization - currently not supported by MPS
        # alpha = 0.001
        # l1_norm = sum(param.abs().sum() for param in CNN_model.parameters())
        # loss_l1 = loss + alpha * l1_norm


        optimizer.zero_grad()            # zero the gradients
        loss.backward()
        # loss_l1.backward()             # for adding regularization
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
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        outputs = CNN_model(X_test)
        loss = loss_function(outputs, y_test)
        test_loss.append(loss.item())

    scheduler.step()


#writer.close()

# Saving the Model
def save_the_model(learning_rate, num_epochs, destination_address, CNN_model, yaml_input):
    """ Saves the model and the yml with parameters that were used """
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    name = 'lr_' + str(learning_rate) + '_epochs_' + str(num_epochs) + '_' + timestr
    filename = destination_address + name + '.pt'
    torch.save(CNN_model, filename)

    import yaml
    # saving the yaml with all the model parameters
    with open(destination_address + name + ".yml", 'w') as file:
        documents = yaml.dump(yaml_input, file)


save_the_model(learning_rate, num_epochs, destination_address, CNN_model, yaml_input)




'''
    ===================================
    Testing
    ===================================
'''

import os
os.system('say "Cookie, I am plotting the picture." ')

y_predicted, accuracies, n_class_correct, n_class_samples = plotting_results.test(CNN_model, X_test, y_test, classes)
plotting_results.plot_image(training_loss, test_loss, num_epochs, learning_rate, classes, accuracies, y_test, y_predicted, filename, show = True)

