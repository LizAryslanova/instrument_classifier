import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim




'''
    ===================================
    Prearing the dataset
    ===================================
'''

import load_kaggle_set


kaggle_folder_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'
kaggle_csv_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'


kaggle_test_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Test_spectrograms/'
kaggle_csv_test_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Test.csv'

classes = ('guitar', 'piano', 'drum', 'violin')


#import data as numpy arrays
#X_train, y_train = load_kaggle_set.process_image_folder(kaggle_folder_address, kaggle_csv_address)
X_test, y_test = load_kaggle_set.process_image_folder(kaggle_test_address, kaggle_csv_test_address)

# print('y_train.shape = ', y_train.shape)


#X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

#y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

#y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

#reshape y tensors
# y_train = y_train.view(y_train.shape[0], 1)
# y_test = y_test.view(y_test.shape[0], 1)

#print(X_train.shape)
#print(y_train.shape)



class TrainSet(Dataset):

    def __init__(self, transform=None):
        # data loading
        X_train, y_train = load_kaggle_set.process_image_folder(kaggle_folder_address, kaggle_csv_address)

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

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 74 * 74, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes) # have 4 classes in the dataset

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


learning_rate = 0.001
num_epochs = 5

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

        # zero gradients
        optimizer.zero_grad()
        # backward
        loss.backward()
        # updates
        optimizer.step()

        if (i+1) % 50 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


'''
    ==============================
    Testing
    ==============================
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

    for i in range(80):
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









'''
o1 = (308 - 5+ 0) / 1 + 1
o2 = o1 // 2
o3 = (o2 - 5 + 0) / 1 + 1
o4 = o3 // 2

print(o1, o2, o3, o4)


o1 = (310 - 5+ 0) / 1 + 1
o2 = o1 // 2
o3 = (o2 - 5 + 0) / 1 + 1
o4 = o3 // 2

print(o1, o2, o3, o4)
'''