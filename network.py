import numpy as np

#Define small random weight matrices and displacement vectors with the proper dimensions
def init_params():
    #w1, 64 x 784 matrix and b1, 64 x 1 vector
    W1 = np.random.normal(0, 0.01, (64, 784))
    b1 = np.random.normal(0, 0.01, (64, 1))
    #w2, 10 x 64 matrix and b2, 10 x 1 vector
    W2 = np.random.normal(0, 0.01, (10, 64))
    b2 = np.random.normal(0, 0.01, (10, 1))
    return W1,b1,W2,b2

#Transform a raw pixel data vector (X) into a probability distribution vector (A2)
def forward_prop(W1, b1, W2, b2, X):
    #1st affine transform
    Z1 = np.matmul(W1, X) + b1
    #Non-linear ReLU function 
    A1 = np.maximum(0, Z1)
    #2nd affine transform
    Z2 = np.matmul(W2, A1) + b2
    #Softmax function to transform elements into probabilities
    A2 = np.exp(Z2) / np.sum(np.exp(Z2), axis=0, keepdims=True)
    return Z1, A1, Z2, A2


#Take an array of true digit labels and turn it into a one-hot vector array (Y)
def labels_translator(labels):
    hot_list = []
    for label in labels:
        one_hot = np.zeros((10, 1))
        np.put(one_hot, label, 1)
        hot_list.append(one_hot)
    Y = np.hstack(hot_list)
    return Y

#Backward pass, mathematical computations for the gradient of the loss function
def loss_gradient(W2, Z1, A1, A2, X, Y):
    #Total number of images in our current batch, value used to average the error
    m = len(Y[0])
    #Output Error (dZ2): raw difference between our predictions (A2) and the true labels (Y). Shape: (10, m)
    dZ2 = A2 - Y
    #Layer 2 weight gradients (dW2): contribution of W2 to the error, calculated as output error times W2's inputs (A1). Shape: (10, 64)
    dW2 = 1/m * np.matmul(dZ2, A1.T)
    #Layer 1 hidden error (dZ1): output error routed back to the Z1 layer using W2 transposed. Shape: (64, m)
    #(Z1 > 0) boolean shortcut representing ReLU derivative. Filters inactive nodes (<0) during the forward pass
    dZ1 = np.matmul(W2.T, dZ2) * (Z1 > 0)
    #Layer 1 weight gradients (dW1): contribution of W1 to the error, calculated as hidden error times input pixels (X). Shape: (64, 784)
    dW1 = 1/m * np.matmul(dZ1, X.T)
    #Bias gradients (db2, db1): sum of the errors for each node, obtained by summing horizontally (axis=1). Same shape as b2, b1.
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dZ2, dW2, dZ1, dW1, db2, db1


#Test
#Init test parameters
X = np.random.rand(784, 6)
W1, b1, W2, b2 = init_params()
labels = [7, 3, 5, 1, 9, 0]
#Math tests
Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
Y = labels_translator(labels)
dZ2, dW2, dZ1, dW1, db2, db1 = loss_gradient(W2, Z1, A1, A2, X, Y)