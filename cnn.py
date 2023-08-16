import numpy as np
import matplotlib.pyplot as plt




def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image,
    as illustrated in Figure 1.

    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions

    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """

    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)))

    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
    of the previous layer.

    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = np.multiply(a_slice_prev, W)
    # Sum over all entries of the volume s.
    Z = np.sum(s)
    # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
    Z = Z + b[0,0,0]

    return Z




'''

     There is an issue with dimentionaliti of the height of the output! !!!!!!!!


'''



def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer,
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    # YOUR CODE STARTS HERE

    # Retrieve dimensions from A_prev's shape (≈1 line)
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparameters['stride']
    pad = hparameters['pad']

    # Compute the dimensions of the CONV output volume using the formula given above.
    # Hint: use int() to apply the 'floor' operation. (≈2 lines)
    n_H = int( (n_H_prev - f - 2 * pad) / stride ) + 1
    n_W = int( (n_W_prev - f - 2 * pad) / stride ) + 1


    print('n_H_prev = ', n_H_prev)
    print('n_W_prev = ', n_W_prev)
    print('n_C = ', n_C)


    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros( (m, n_H, n_W, n_C))

    print('Z.shape = ', Z.shape)

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):               # loop over the batch of training examples
        a_prev_pad = A_prev[i]               # Select ith training example's padded activation

        print('a_prev_pad.shape = ', a_prev_pad.shape)

        for h in range(n_H):           # loop over vertical axis of the output volume
            # Find the vertical start and end of the current "slice"
            vert_start = h
            vert_end = h + stride + 1


            for w in range(n_W):       # loop over horizontal axis of the output volume
                # Find the horizontal start and end of the current "slice"
                horiz_start = w
                horiz_end = w + stride + 1

                for c in range(n_C):   # loop over channels (= #filters) of the output volume

                    print('i = ', i, 'h = ', h, 'w = ', w, 'c = ', c)

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = a_prev_pad[ vert_start:vert_end, horiz_start:horiz_end, : ]

                    print('Im here 1')

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈3 line)
                    weights = W[:,:,:,c]

                    print('Im here 2')
                    biases = b[:,:,:,c]

                    print('Im here 3')

                    print('a_slice_prev.shape = ', a_slice_prev.shape)
                    print('weights.shape = ', weights.shape)
                    print('biases.shape = ', biases.shape)



                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

                    print('Z.shape = ', Z.shape)

                    print('Im here 4')

    # YOUR CODE ENDS HERE

    # Save information in "cache" for the backprop
    cache = (A_prev, W, b, hparameters)

    return Z, cache