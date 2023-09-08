# importing Numpy package
import numpy as np



# how to copy part of the np.array

X = np.zeros(shape=(10, 4, 3, 2))
print(X.shape)

copy = np.zeros(shape=(6, 4, 3, 2))
copy = X[:6]
print(copy.shape)