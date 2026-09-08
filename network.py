import numpy as np

def init_params():
    #We want to take a 784 x 1 pixel representation vector input and scale it down to 10 x 1 with affine transforms and non-linear functions
    #w1, 64 x 784 matrix and b1, 64 x 1 vector
    W1 = np.random.normal(0, 0.01, (64, 784))
    b1 = np.random.normal(0, 0.01, (64, 1))
    #w2, 10 x 64 matrix and b2, 10 x 1 vector
    W2 = np.random.normal(0, 0.01, (10, 64))
    b2 = np.random.normal(0, 0.01, (10, 1))
    return W1,b1,W2,b2

def forward_prop(W1, b1, W2, b2, X):
    #1st affine transform
    Z1 = np.matmul(W1, X) + b1
    #Non-linear ReLU function 
    A1 = np.maximum(0, Z1)
    #2nd affine transform
    Z2 = np.matmul(W2, A1) + b2
    #Softmax function to transform elements into probabilities
    A2 = np.exp(Z2) / np.sum(np.exp(Z2))
    return A2

def labels_translator(labels):
    #We take an array of true lables and turn it into one-hot vectors
    hot_list = []
    for label in labels:
        one_hot = np.zeros((10, 1))
        np.put(one_hot, label, 1)
        hot_list.append(one_hot)
    hot_arr = np.hstack(hot_list)
    return hot_arr

#Test, should output 1.0
X = np.random.rand(784, 1)
(W1, b1, W2, b2) = init_params()
print(np.sum(forward_prop(W1, b1, W2, b2, X)))

#Test2
labels = [7, 3, 5, 1, 9, 0]
print(labels_translator(labels))