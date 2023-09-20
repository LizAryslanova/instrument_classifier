import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

import utils



'''
    ===================================
    Prearing the dataset
    ===================================
'''


classes = ('guitar', 'piano', 'drum', 'violin')


# Un-Pickle Test sets
with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_test_x', 'rb') as f:
    X_test = pickle.load(f)
with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_test_y', 'rb') as f:
    y_test = pickle.load(f)


X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test).type(torch.LongTensor)




class TrainSet(Dataset):

    def __init__(self, transform=None):
        # data loading
        with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_train_x', 'rb') as f:
            X_train = pickle.load(f)
        with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_train_y', 'rb') as f:
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
    CNN model
    ===================================
'''

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)     # 1 input channel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * utils.dimensions_for_linear_layer(X_test.shape[2], X_test.shape[3]), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



'''
    ===================================
    Hyperparameters
    ===================================
'''

num_classes = 4
CNN_model = CNN()

learning_rate = 0.00001
num_epochs = 2

loss_function = nn.CrossEntropyLoss()  # softmax is included
optimizer = optim.SGD(CNN_model.parameters(), lr = learning_rate)

destination_address = '/Users/cookie/dev/instrumant_classifier/model_results/'



'''
    ===================================
    Training loop
    ===================================
'''

n_total_steps = len(train_loader)
training_loss = []
test_loss = []

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):
        # forward pass and loss
        y_predicted = CNN_model(images)

        loss = loss_function(y_predicted, labels)

        optimizer.zero_grad()     # zero gradients
        loss.backward()
        optimizer.step()                 # updates

        if (i+1) % 50 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    training_loss.append(loss.item())

    # Calculating the test loss
    with torch.no_grad():
        outputs = CNN_model(X_test)

        loss = loss_function(outputs, y_test)
        test_loss.append(loss.item())



# Saving the Model
# torch.save(CNN_model, '/Users/cookie/dev/instrumant_classifier/unit_testing/cnn_for_tests.pt')




#import os
#os.system('say "Cookie, I am plotting losses" ')






'''
# plotting losses
plt.plot(training_loss, 'g', label = 'Training loss')
plt.plot(test_loss, 'r', label = 'Test loss')

plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.title('Loss functions \n LR = str(learning_rate), epochs = str(num_epochs)')

plt.title(f'Loss functions for {num_epochs} epochs \n Learnind rate = {learning_rate}, Accuracy = ')

plt.legend()
plt.show()
'''


'''
    ===================================
    Testing
    ===================================
'''

import os
os.system('say "Cookie, Im plotting the picture." ')

y_predicted, accuracies, n_class_correct, n_class_samples = utils.test(CNN_model, X_test, y_test, classes)

utils.plot_image(training_loss, test_loss, num_epochs, learning_rate, classes, accuracies, y_test, y_predicted, destination_address, show = True)



import os
os.system('say "Cookie, your program has finished." ')

