import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim

import pickle

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



# Calculate dimensions for the nn.Linear layer
o_01 = (X_test.shape[2] - 5 + 0) / 1 + 1
o_02 = o_01 // 2
o_03 = (o_02 - 5 + 0) / 1 + 1
output_shape_1 = o_03 // 2

o_11 = (X_test.shape[3] - 5 + 0) / 1 + 1
o_12 = o_11 // 2
o_13 = (o_12 - 5 + 0) / 1 + 1
output_shape_2  = o_13 // 2
# print(output_shape_1, output_shape_2)


X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test)
y_test = torch.tensor(y_test, dtype=torch.long)



class TrainSet(Dataset):

    def __init__(self, transform=None):
        # data loading
        with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_train_x', 'rb') as f:
            X_train = pickle.load(f)
        with open('/Users/cookie/dev/instrumant_classifier/pickles/kaggle_train_y', 'rb') as f:
            y_train = pickle.load(f)

        self.x = torch.from_numpy(X_train.astype(np.float32))
        y_train = torch.from_numpy(y_train)
        self.y = torch.tensor(y_train, dtype=torch.long)
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
        self.fc1 = nn.Linear(16 * int(output_shape_1) * int(output_shape_2), 120)
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

learning_rate = 0.005
num_epochs = 4

loss_function = nn.CrossEntropyLoss()  # softmax is included
optimizer = optim.SGD(CNN_model.parameters(), lr = learning_rate)



'''
    ===================================
    Training loop
    ===================================
'''

n_total_steps = len(train_loader)

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


'''
    ===================================
    Testing
    ===================================
'''

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]


    outputs = CNN_model(X_test)
    # max returns (value ,index)
    _, predicted = torch.max(outputs, 1)
    n_samples += y_test.size(0)
    n_correct += (predicted == y_test).sum().item()

    print('Predicted = ', predicted)

    for i in range(64):   # number of files in the test set
        label = y_test[i]
        pred = predicted[i]
        if (label == pred):
            n_class_correct[label] += 1
        n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(num_classes):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')
        print('n_class_correct = ', n_class_correct, ' n_class_samples = ', n_class_samples)





import os
os.system('say "your program has finished" ')







