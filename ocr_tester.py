import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

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

#Open the handwritten digit image and modify it to match MNIST dataset image format
img = Image.open("number.png")
img = img.convert("L")
img = ImageOps.invert(img)
img = img.resize((28,28))
#Use NumPY to arrange the image as properly-shaped normalized input for the network
X = np.array(img)
X_test = X.reshape((784, 1)) / 255

#Fetch the trained model parameters from the .npz file
model = np.load("trained_model.npz")
W1 = model["W1"]
b1 = model["b1"]
W2 = model["W2"]
b2 = model["b2"]

'''
#DEBUGGING TOOL
plt.imshow(X_test.reshape((28, 28)), cmap="gray")
plt.title("What the Neural Network Actually Sees")
plt.show()
'''

#Apply the forward propagation and print the trained network's guessed digit and confidence
Z1, A1, A2 = forward_prop(W1, b1, W2, b2, X_test)
guess = np.argmax(A2)
confidence = A2[guess, 0] * 100
print(f"The written digit is: {guess}")
print(f"Network Confidence: {confidence:.2f}%")