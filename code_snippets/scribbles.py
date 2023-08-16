import numpy as np


y = np.zeros((2, 3, 4))
print(y.shape)

(a, b, c) = y.shape
print(c)

print(y[:,:,2])