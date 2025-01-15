# Dataset
Obtain dataset used in this project was from [ml.chenghock.com/nn/MNIST_CSV.zip](https://ml.chenghock.com/nn/MNIST_CSV.zip) (citation: [ml.chenghock.com/nn/cite](https://ml.chenghock.com/neuralnetwork/cite))

Taken from the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) containing 28x28 pixel images of handwritten digits.

Images are flatten into a single dimensional array with left-to-right and top-to-bottom directions.

Test size: 

# Features
This is probably not a very efficient implementation of a neural network, but was done to improve my very own understanding of neural networks.

Neural networks are the building blocks of almost all machine learning applications, such as long short term memory recurrent neural networks and convolutional neural networks.

- Supports batch size

# Improvements
- Use `sklearn.sparse_matrix` for performance efficient (stores sparse matrix in a more memory efficient manner)
- Implement Stochastic Gradient Descent instead of regular gradient descent
- Implement a convolutional layer and dropout to improve efficiency and reduce complexity
