import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# Prepare data

X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features = 1, noise = 20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)    # built PyTorch method that will reshape our tensor

n_samples, n_features = X.shape


# 1. Model

input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)


# 2. Loss, optimiser

learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# 3. training loop

num_epochs = 100

for epoch in range(num_epochs):
    # forward, loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward
    loss.backward()

    #update
    optimizer.step()    #updates the weights

    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


# plot
predicted = model(X).detach().numpy()      # detatch - prevents from being tracked
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

