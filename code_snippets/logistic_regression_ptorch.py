import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Prep data

bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target

n_samples, n_features = X.shape

print('n_samples =', n_samples, ' n_features =', n_features)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale features
sc = StandardScaler()
X_train - sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


#reshape y tensors

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# 1. Model
# f = wx + b, sigmoid

class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)



# 2. Loss, optimizer

learning_rate = 0.01
criterion = nn.BCELoss()  #binary cross entropy loss
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3. Training loop

num_epochs = 300

for epoch in range(num_epochs):
    # forward pass and loss

    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    # backward
    loss.backward()

    # updates
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()


    if (epoch+1) % 100 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    accuracy = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {accuracy:.4f}')



