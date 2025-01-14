import NN

import numpy as np

if __name__ == "__main__":
	network = NN.NeuralNetwork([
		NN.InputLayer(3),
		NN.DenseLayer(6),
		NN.DenseLayer(1)
	])

	network.compile()

	x = np.array([[.5, .4, .5]])
	y = np.array([1])
	print("outcome", network.fit(x, y))