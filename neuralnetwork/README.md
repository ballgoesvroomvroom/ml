# Writeup
As part of my learning process, I wrote up an explanation article solving the most challenging problem I encountered.

Read here: [ml.chenghock.com/nn/backpropagation.pdf](https://ml.chenghock.com/nn/backpropagation.pdf)

# About
This is probably not a very efficient implementation of a neural network, but was done to improve my very own understanding of neural networks.

Neural networks are the building blocks of almost all machine learning applications, such as LSTMs and transformer models.

Runs purely off `numpy` and `math` library.

# Challenges
The concept of backpropagation was difficult to implement. Obtaining the derivatives of the loss function with respect to the individual weights was not easy, given how the network architecture can change.

The [article](https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65#6aba) by Omar Aflak helped introduce me to a clever approach to backpropagation (usage covered in my writeup).

# Future Improvements
- Use `sklearn.sparse_matrix` for performance efficient (stores sparse matrix in a more memory efficient manner)
- Implement Stochastic Gradient Descent instead of regular gradient descent
- Implement other form of layers (e.g. dropout layer)
