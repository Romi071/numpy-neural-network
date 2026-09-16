# Neural Network from scratch: MNIST Classifier

This is an educational machine learning project that builds a Fully Connected (Dense) Neural Network entirely from scratch using Python and NumPy. This project intentionally bypasses high-level libraries like TensorFlow or PyTorch to explore the underlying calculus, linear algebra, and backpropagation mechanics of Artificial Intelligence.


## Dataset Setup:
This neural network is designed for the MNIST handwritten digit dataset.
1. Download the 4 raw `.ubyte` files from the following URL: https://www.kaggle.com/datasets/hojjatk/mnist-dataset
2. Place the downloaded files in the same root directory as the script.
3. Run the training script (`network.py`) to train the model and generate the `trained_model.npz` parameters file.
4. Optionally, draw a digit using Paint, save it as `number.png` in the same directory as the script and run the `ocr_tester.py` script to test the model.


## Architecture & Performance
*   **Structure:** 784 Input Nodes -> 64 Hidden Nodes (ReLU Activation) -> 10 Output Nodes (Softmax).
*   **Optimization:** Custom-implemented Gradient Descent and mathematical loss gradients.
*   **Accuracy:** The model successfully converges and consistently achieves **~97%+ accuracy** on the 10,000-image MNIST test set.


## Limitations:
While the model performs exceptionally well on the standardized MNIST test set, it has been shown to struggle with external handwritten user input (such as digits drawn in Paint). This perfectly highlights the fundamental limitations of Dense Neural Network architectures in computer vision:
*   **Spatial Rigidity:** Dense networks rely on exact pixel coordinate weights. A stroke that is slightly thicker, tilted, or shifted by just about two pixels triggers entirely different input nodes, instantly breaking the model's guess.
*   **Translation Invariance:** Because fully connected layers look at the whole canvas at once, they cannot scan an image for localized features.
*   **Distribution Shift:** The network heavily overfit to the exact stroke dynamics found in the training dataset, failing to recognize the "shape" of an external input.


## Next Steps
This project served as a foundational mathematics and machine learning exercise. To achieve real-world image processing capabilities that overcome the model's inherent spatial limitations, the natural next step is to implement a Convolutional Neural Network (CNN).