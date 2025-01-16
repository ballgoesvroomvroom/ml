"""
custom implementation of neural network and neurons, with activiation functions
with feedforward and backpropagation (with gradient descent) implementation

from here onwards, the term "network" is the short form of "neural networks"
"""
import math
import numpy as np

class Layer():
	"""
	contains neurons or activation function
	"""
	def __init__(self, neuron_count, activation=""):
		self.neuron_count = neuron_count

		# vectorize activation function so it can be directly applied to our layer outcome in the form of numpy.ndarray
		if (activation == "sigmoid"):
			self.activation = np.vectorize(lambda x: 1 /(1 +math.exp(-x)))
			self.activation_prime = np.vectorize(lambda x: 1 /(1 +math.exp(-x)) *(1 -1 /(1 +math.exp(-x))))
		elif (activation == "relu"):
			self.activation = np.vectorize(lambda x: max(0, x))
			self.activation_prime = np.vectorize(lambda x: 1 if x >= 0 else 0)

		
class DenseLayer(Layer):
	def build(self, input_size):
		"""
		input_size: integer, number of neurons (or inputs) that connects the previous layer)
		"""
		self.input_size = input_size # store input size (i), useful for calculating derivative of loss function with respect to individual weights

		# initialise weights with He normal initialisation method
		weights = np.array([np.random.randn(input_size) *np.sqrt(1 /input_size) for i in range(self.neuron_count)]) # shape = (j, i), where i represents number of input neurons from previous layer, j represents number of output neurons this layer has

		self.weights = weights
		self.biases = np.zeros(self.neuron_count) # initialise bias with value of 0

	def forward(self, input_feature):
		"""
		input_feature: one-D numpy.ndarray with .ndim == 1
		computes the output of the this layer

		1. computes dot product of previous input with individual connection weights to obtain each neuron output
		2. add neuron bias to each neuron output
		3. applies activation function to each neuron (if set)

		returns the activation of this layer
		"""
		mul_outcome = np.matmul(input_feature, self.weights.T) # (input_feature_size,) * (input_feature_size, neuron_count) = (neuron_count,)
		add_outcome = mul_outcome +self.biases # mul_outcome and biases have the same shape

		if (hasattr(self, "activation")):
			return self.activation(add_outcome)
		return add_outcome

	def backward(self, output, activations, loss, learning_rate):
		"""
		output: Y, one-D numpy.ndarray with .ndim == 1, output for this layer (Y1, ..., Yj, where j represents number of output neurons for this layer)
		activations: X, one-D numpy.ndarray with .ndim == 1, input for this layer (X1, ..., Xi, where i represents number of input neurons from previous layer)
		loss: dE/dY(j), one-D numpy.ndarray with .ndim == 1, derivative of loss function with respect to the output of each neuron (Y1, ..., Yj, where j represents number of output neurons)
		learning_rate: float, neural network's learning rate

		1. computes derivative of loss function with respect to the individual weights
		2. uses the current value (slope) to adjust the weights in this layer (with respect to learning_rat`e)
		3. computes derivative of loss function with respect to the individual biases
		4. uses the current value (slope) to adjust the biases in this layer (with respect to learning_rate)
		5. computes the derivative of loss function with respect to the inputs received by this layer
		6. propogates the current value (slope) backwards into the next traversed layer (to be used as derivative of loss function with respect to the output of each neuron)

		returns the current value of the derivative of loss function with respect to the layer inputs
		"""
		# compute dE/dW(j)(i), where W = [wj1, ..., wji] and i represents number of input neurons (derivative of loss function with respect to weights --> our main goal! because we want to find the value of the individual rates where loss function is lowest (where rate of change = 0))
		# w(j)(i), means the ith weight in the jth neuron of the current layer (a neuron has i amount of weights because there are i amount of inputs - FC layer); W shares same shape as self.weights
		# from the chain rule, dE/dW(j)(i) = dE/dY(j) * dY(j)/dW(j)(i)
		# dY(j)/dW(j)(i) = Xi, where X is the input array (this is because Y = XW + B, where Y is the output, X is the input, and W is the weights, dot product is used here)
		loss = loss.reshape((1, self.neuron_count)) # inflate to 2D array for dot product operation
		activations = activations.reshape((1, self.input_size)) # inflate to 2D array for dot product operation
		weights_grad = np.matmul(loss.T, activations) # dE/dW(j)(i), we have accomplished our main challenge in backpropagation
		if (hasattr(self, "activation")):
			# dY(j)/dW(j)(i) = activation_prime(Y(j)) *Xi
			# element-wise multiply activation_prime(Y(j)) row based into weights_grad
			weights_grad = np.multiply(weights_grad, self.activation_prime(output).reshape((self.neuron_count, 1)))

		# compute dE/dB(j), where B = [b1, ..., bj] and j represents number of output neurons
		# from chain rule, dE/dB(j) = dE/dY(j) * dY(j)/dB(j)
		# dY(j)/dB(j) = 1 (this is because Y = XW + B, hence dY/dB = 1)
		# effectively, dE/dB(j) = dE/dY(j) = loss
		bias_grad = loss

		# adjust weights and biases accordingly
		self.weights -= weights_grad *learning_rate
		self.biases -= bias_grad.flatten() *learning_rate

		# compute dE/dX(i) to be used as the loss function (dE/dY(j), where j is number of output neurons for the previous layer) for the previous layer (the layer traverse to next, because we are traversing backwards for backprop)
		# dE/dX(i) = dE/dY(1) * dY(1)/dX(i) + ... + dE/dY(j) * dY(j)/dX(i), where j represents number of output neurons for previous layer
		# in a FC layer, X(i) is connected to every output neuron (Y(j)), hence it will also contribute to the loss function. that is why the we iterate through each output neuron (can be done in a single operation with dot products)
		# dY(j)/dX(i) = W(j)(i), ith weight in jth neuron of the current layer
		if (hasattr(self, "activation")):
			# dY(j)/dX(i) = activation_prime(Y(j)) * w(j)(i)
			output_change_respect_input = self.activation_prime(output).reshape((self.neuron_count, 1)) # shape = (j, 1)
			output_change_respect_input = np.multiply(self.weights, output_change_respect_input)
		else:
			# dY(j)/dX(i) is simply w(j)(i)
			output_change_respect_input = self.weights
		input_grad = np.matmul(loss, output_change_respect_input)
		return input_grad # chain this return value to be passed as the loss argument (dE/dY(j)) for the next traversed layer

class InputLayer(Layer):
	"""
	first layer in network
	no weights and biases applied
	"""
	def build(self, input_size):
		"""
		no need to apply weights and biases
		"""
		pass

	def forward(self, input_feature):
		"""
		no weights and biases to apply function to
		"""
		return input_feature

class NeuralNetwork():
	"""
	contains layers of neuron and activation function
	"""
	def __init__(self, layers, learning_rate=0.02):
		"""
		layers: Layer[], contains the layers for the network, first and last element contains the input and output layer; therefore, min length for layers is 2 
		learning_rate: float, influences step size when learning weights
		"""
		self.layers = layers
		self.learning_rate = learning_rate

	def compile(self):
		"""
		model the network with its layers and neurons
		prepare it for learning
		"""
		prev_layer_shape = None # store currently iterated layer output size (i.e. number of neurons), not needed for first layer (InputLayer)
		for layer in self.layers:
			layer.build(prev_layer_shape)
			prev_layer_shape = layer.neuron_count


	def fit(self, x, y):
		"""
		x: numpy.ndarray with shape (num_of_observations, num_of_features)
		y: numpy.ndarray with shape (num_of_output_labels,)

		num_of_features should correspond to number of neurons in self.layers[0].neurons

		fits the network with the provided data
		performs backpropagation with gradient descent to learn the weights
		"""
		# 250 epochs
		for epoch in range(500):
			total_loss = 0 # use mean squared residuals (mse) as loss function
			predictions = []

			for sample_idx, sample in enumerate(x):
				# feedforward
				activations = [] # store activations to help compute loss function derivatives with respect to the individual weights
				for layer in self.layers:
					if (len(activations) >= 1):
						# subsequent layer, pass in output from previous layer as the input to this current layer
						activations.append(layer.forward(activations[-1]))
					else:
						# first layer, pass in input
						activations.append(layer.forward(sample))

				# compute total loss
				# activations[-1] stores the output of the final layer, aka prediction output
				loss = ((y[sample_idx] -activations[-1]) **2).mean() # mse, 1/n(Y(j)^ - Y(j))**2, where Y(j)^ is the ground truth
				total_loss += loss

				# E = 1/n * ((Y(1)^ - Y(1))**2 + ... + (Y(j)^ - Y(j))**2) --> MSE function
				# compute dE/dY(j) --> derivative of loss function (E) with respect to Y(j), (dE/dY(j), where j represents index of neuron in output layer)
				loss_grad = (2 /activations[-1].size) *activations[-1] -y[sample_idx] # dE/dY(j) = 2/n(Y(j) - Y(j)^), where Y(j)^ is the ground truth

				# backpropagation
				# start from outer most layer - output layer first (because dE/dX for the subsequent layer can be computed by the current dE/dY for the current layer, where X is the input and Y is the output)
				for i in range(len(self.layers) -1, 0, -1): # dont traverse input layer (first layer) since there are no weights there
					loss_grad = self.layers[i].backward(activations[i], activations[i -1], loss_grad, self.learning_rate) # obtain dE/dX(i) for current layer, to be used as dE/dY(j) for next traversed layer

				# store prediction for current sample
				predictions.append(activations[-1])

			# compute average loss
			print("avg loss", total_loss /y.size)

		# todo: implement backpropagation
		return predictions




