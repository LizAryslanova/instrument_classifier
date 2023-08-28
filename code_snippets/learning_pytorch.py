import torch


x = torch.empty(4, 2)
print('x = ', x)



y = torch.rand(2, 3)
print('y = ', y)

z = torch.rand(2, 3)

y.add_(z)
print('y = ', y)

