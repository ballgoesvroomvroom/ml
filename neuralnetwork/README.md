# Dataset
Obtain dataset used in this project was from [ml.chenghock.com/nn/MNIST_CSV.zip](https://ml.chenghock.com/nn/MNIST_CSV.zip) (citation: [ml.chenghock.com/nn/cite](https://ml.chenghock.com/nn/cite))

Taken from the [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) containing 28x28 pixel images of handwritten digits.

Images are flatten into a single dimensional array with left-to-right and top-to-bottom directions.

Test size: 

# Features
This is probably not a very efficient implementation of a neural network, but was done to improve my very own understanding of neural networks.

Neural networks are the building blocks of almost all machine learning applications, such as long short term memory recurrent neural networks and convolutional neural networks.

- Supports batch size

# Challenges
The concept of backpropagation was difficult to implement. Obtaining the derivatives of the loss function with respect to the individual weights was not easy, given how the network architecture can change.

The [article](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65#6aba) by Omar Aflak helped introduce me to a clever approach to backpropagation.

# Improvements
- Use `sklearn.sparse_matrix` for performance efficient (stores sparse matrix in a more memory efficient manner)
- Implement Stochastic Gradient Descent instead of regular gradient descent
- Implement a convolutional layer and dropout to improve efficiency and reduce complexity
