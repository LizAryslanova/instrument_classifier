import numpy as np
import pickle

# Create a Numpy array
array = np.arange(10)
print('array = ', array)

# Save the array with Pickle
with open('/Users/cookie/dev/instrumant_classifier/code_snippets/array.pkl', 'wb') as f:
    pickle.dump(array, f)






# Load the array with Pickle
with open('/Users/cookie/dev/instrumant_classifier/code_snippets/array.pkl', 'rb') as f:
    loaded_array = pickle.load(f)

print('loaded_array = ', loaded_array)