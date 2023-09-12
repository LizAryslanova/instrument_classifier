import numpy as np

# Create a NumPy array of complex numbers
complex_array = np.array([1+2j, 3+4j, 5+6j])


# Compute the norm of the complex numbers
norms = np.linalg.norm(complex_array)

# Normalize the complex numbers
normalized_array = complex_array / norms


print(complex_array)

print('============')

print(normalized_array)