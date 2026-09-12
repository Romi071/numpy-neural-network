import numpy as np
import matplotlib.pyplot as plt

#Define small random weight matrices and displacement vectors with the proper dimensions
def init_params():
    #w1, 64 x 784 matrix and b1, 64 x 1 vector
    W1 = np.random.normal(0, 0.01, (64, 784))
    b1 = np.random.normal(0, 0.01, (64, 1))
    #w2, 10 x 64 matrix and b2, 10 x 1 vector
    W2 = np.random.normal(0, 0.01, (10, 64))
    b2 = np.random.normal(0, 0.01, (10, 1))
    return W1,b1,W2,b2

#Take an array of true digit labels and turn it into a one-hot vector array (Y)
def labels_translator(labels):
    hot_list = []
    for label in labels:
        one_hot = np.zeros((10, 1))
        np.put(one_hot, label, 1)
        hot_list.append(one_hot)
    Y = np.hstack(hot_list)
    return Y

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
    return Z1, A1, A2

#Backward pass, mathematical computations for the weight gradients of the loss function
def loss_gradient(W2, Z1, A1, A2, X, Y):
    #Total number of images in our current batch, value used to average the error
    m = len(Y[0])
    #Output Error (dZ2): raw difference between our predictions (A2) and the true labels (Y). Shape: (10, m)
    dZ2 = A2 - Y
    #Layer 2 weight gradients (dW2): contribution of W2 to the error, calculated as output error times W2's inputs (A1). Shape: (10, 64)
    dW2 = 1/m * np.matmul(dZ2, A1.T)
    #Layer 1 hidden error (dZ1): output error routed back to the Z1 layer (higher dim.) using W2 transposed. Shape: (64, m)
    #(Z1 > 0) boolean shortcut representing ReLU derivative. Filters inactive nodes (<0) during the forward pass
    dZ1 = np.matmul(W2.T, dZ2) * (Z1 > 0)
    #Layer 1 weight gradients (dW1): contribution of W1 to the error, calculated as hidden error (dZ1) times input pixels (X). Shape: (64, 784)
    dW1 = 1/m * np.matmul(dZ1, X.T)
    #Bias gradients (db2, db1): sum of the errors for each node, obtained by summing horizontally (axis=1). Same shape as b2, b1.
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    return dW2, dW1, db2, db1

#Update the parameters in the direction of the gradient times a factor alpha
def update_params(W1, dW1, b1, db1, W2, dW2, b2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

#Train the network with a given database, X input is taken to be a massive data matrix
def train_network(X, labels, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    Y = labels_translator(labels)
    success_rate = []
    #Training loop
    for _ in range(iterations):
        #Slice X, Y into 1000 batches and go over them to train more effectively 
        for k in range(1000):
            Xi = X[:, k * X.shape[1] // 1000 : (k + 1) * X.shape[1] // 1000]
            Yi = Y[:, k * Y.shape[1] // 1000 : (k + 1) * Y.shape[1] // 1000]
            #Obtain the prob distribution, loss gradient and update the parameters accordingly
            Z1, A1, A2 = forward_prop(W1, b1, W2, b2, Xi)
            dW2, dW1, db2, db1 = loss_gradient(W2, Z1, A1, A2, Xi, Yi)
            W1, b1, W2, b2 = update_params(W1, dW1, b1, db1, W2, dW2, b2, db2, alpha)
            #Determine the networks's mean success rate for this batch
            guess = np.argmax(A2, axis=0)
            correct = np.array(labels[k * len(labels) // 1000 : (k+1) * len(labels) // 1000])
            success_rate.append(np.mean(guess == correct))
        #Print the mean success rate for every batch in this iteration to visualize learning
        print(f"The network's success rate for this iteration was: {100 * np.mean(success_rate)}%")
        success_rate = []
    print(f"💪 Training Completed!")
    return W1, b1, W2, b2


#Fetch the training data pixels from a binary (ubyte) file, reshape it into a 2D array and normalize it:
with open("train-images.idx3-ubyte", "rb") as file:
    binary_data = file.read()
    train_data = np.frombuffer(binary_data, dtype=np.uint8, offset=16)
    X = train_data.reshape((60000, 784)).T / 255
#Fetch the training data labels from a binary (ubyte) file:
with open("train-labels.idx1-ubyte", "rb") as file:
    binary_labels = file.read()
    train_labels = np.frombuffer(binary_labels, dtype=np.uint8, offset=8)


"""
#Optional binary image plotter (change the integer index to check other images)
print(train_labels[5687])
plt.imshow(((X[:, 5687]).reshape((28, 28))))
plt.show()
"""

#Set parameters and train
alpha = 0.01
iterations = 100
W1, b1, W2, b2 = train_network(X, train_labels, alpha, iterations)

#Fetch the test data pixels from a binary (ubyte) file, reshape it into a 2D array and normalize it:
with open("t10k-images.idx3-ubyte", "rb") as file:
    binary_test_data = file.read()
    test_data = np.frombuffer(binary_test_data, dtype=np.uint8, offset=16)
    X_test = test_data.reshape((10000, 784)).T / 255
#Fetch the test data labels from a binary (ubyte) file:
with open("t10k-labels.idx1-ubyte", "rb") as file:
    binary_test_labels = file.read()
    test_labels = np.frombuffer(binary_test_labels, dtype=np.uint8, offset=8)

#Perform the test using the trained params on 10k provided new images and print success rate
Z1, A1, A2 = forward_prop(W1, b1, W2, b2, X_test)
guess = np.argmax(A2, axis=0)
correct_rate = np.mean(guess == test_labels)
print(f"✅ The trained network's correct rate for this test data was: {100 * correct_rate}%")