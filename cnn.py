import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim



# Propping the dataset

import load_kaggle_set



kaggle_folder_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Train_spectrograms/'
kaggle_csv_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Train.csv'


kaggle_test_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Test_spectrograms/'
kaggle_csv_test_address = '/Users/cookie/dev/instrumant_classifier/audio_files/from_kaggle/Metadata_Test.csv'


#import data as numpy arrays
X_train, y_train = load_kaggle_set.process_image_folder(kaggle_folder_address, kaggle_csv_address)
X_test, y_test = load_kaggle_set.process_image_folder(kaggle_test_address, kaggle_csv_test_address)

print('y_train.shape = ', y_train.shape)


X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

#reshape y tensors
#y_train = y_train.view(y_train.shape[0], 1)
#y_test = y_test.view(y_test.shape[0], 1)

print(X_train.shape)
print(y_train.shape)





class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 74 * 74, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4) # have 4 classes in the dataset

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



CNN_model = CNN()
learning_rate = 0.01

loss_function = nn.CrossEntropyLoss()  # softmax is included
optimizer = optim.SGD(CNN_model.parameters(), lr = learning_rate)


# training loop

num_epochs = 1

for epoch in range(num_epochs):
    # forward pass and loss

    y_predicted = CNN_model(X_train)

    print('y_predicted.shape = ', y_predicted.shape)


    loss = loss_function(y_predicted, y_train)

    # zero gradients
    optimizer.zero_grad()

    # backward
    loss.backward()

    # updates
    optimizer.step()



    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


with torch.no_grad():
    y_predicted = CNN_model(X_test)
    y_predicted_cls = y_predicted.round()
    accuracy = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {accuracy.item():.4f}')
















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