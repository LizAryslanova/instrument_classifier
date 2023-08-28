import torch


print('Creating tensors')

x = torch.empty(4, 2)
print('x = ', x)

print(x[1, :])
print(x[1,1])

# for tensors with only one element. to get the actual value
print(x[1,1].item())


y = torch.rand(2, 3)
print('y = ', y)

z = torch.rand(2, 3)

print('=============================')

print('Simple operations')


# ADDITION (elementwise)
q = y + z
q = torch.add(y,z)
y.add_(z)   # trailing underscore will do an inplace operation. (modify the thing you apply it to)
print('y = ', y)

# SUBSTRACTION
q = y - z
q = torch.sub(y, z)

# MULTIPLICATION
q = torch.mul(y,z)
y.mul_(z)

# DIVISION
q = y/z
q = torch.div(y,z)


print('=============================')

print('Slicing and reshaping')

# SLICING OPERATIONS

x = torch.rand(5, 3)
print(x)

print(x[:, 0])  # all rows, first column
print(x[1, :])  # second row and all columns



# RESHAPING

x = torch.rand(4, 4)
print(x)

y = x.view(16)
print(y)

z = x.view(-1, 8)
print(z)




# NB! If items are stored on CPU (not GPU) they share memory location, so by modifying one you will modify them both
# Converting from tensors to numpy.

import numpy as np


a = torch.ones(5)
print(a)

b = a.numpy()
print(b)
print(type(b))


# numpy to tensor

a = np.ones(5)
print(a)
b = torch.from_numpy(a) # can add dtype = ... to modify data type
print(b)




# For calculating gradients
x = torch.ones(5, requires_grad=True)




# Autograd - calculating gradients
# calculates using vector Jacobian products / chain rule

print('=============================')
print('Calculating gradients')

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2

print(y)

z = y * y * 2
print('z = ', z)


z = z.mean()
print('z mean = ', z)

z.backward()  #dz/dx   # empty brackets because z is scalar
print(x.grad)


print('=============================')

# For vector inputs

x = torch.randn(3, requires_grad=True)
print(x)

y = x + 2

print(y)

z = y * y * 2
print('z = ', z)


v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)
print(x.grad)



print('=============================')


# To exclude an operation for gradient computation

#1.  x.requires_grad_(False)
#2.  y = x.detatch()
#3. with torch.no_grad():
#       y = x+2
#       print(y)




# NB gradients accumulate

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights * 3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()