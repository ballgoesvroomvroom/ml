import NN

import numpy as np

if __name__ == "__main__":
	network = NN.NeuralNetwork([
		NN.InputLayer(2),
		NN.DenseLayer(3, activation="relu"),
		NN.DenseLayer(3, activation="relu"),
		NN.DenseLayer(3, activation="relu"),
		NN.DenseLayer(3),
		NN.DenseLayer(1, activation="relu")
	])

	network.compile()

	x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	y = np.array([[0], [1], [1], [0]])
	print("outcome", network.fit(x, y))
	print("truth", y)

	print(network.layers[2].weights)