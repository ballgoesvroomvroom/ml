"""
custom implementation of neural network and neurons, with activiation functions
with feedforward and backpropagation (with gradient descent) implementation

from here onwards, the term "network" is the short form of "neural networks"
"""
import math
import random
import numpy as np

class Neuron():
	"""
	contains weights and a single bias
	"""
	def __init__(self, weights, bias):
		"""
		weights: numpy.ndarray with shape (input_feature_size,)
		bias: float, single bias
		"""
		self.weights = weights
		self.bias = bias

class Layer():
	"""
	contains neurons or activation function
	"""
	def __init__(self, neuron_count, activation=""):
		self.neuron_count = neuron_count

		# vectorize activation function so it can be directly applied to our layer outcome in the form of numpy.ndarray
		if (activation == "sigmoid"):
			self.activation = np.vectorize(lambda x: 1 /(1 +math.exp(-x)))
		elif (activation == "relu"):
			self.activation = np.vectorize(lambda x: max(0, x))

	def build(self, input_size):
		"""
		input_size: integer, number of neurons (or inputs) that connects the previous layer)
		"""
		# initialise random weights
		weights = np.random.randn(input_size) *np.sqrt(1 /input_size)
		# weights = np.ones(input_size)

		# initialise with bias set to 0
		bias = 0

		self.neurons = [Neuron(weights, bias) for x in range(self.neuron_count)]

	def outcome(self, input_feature):
		"""
		input_feature: one-D numpy.ndarray with .ndim == 1
		computes the output of the this layer

		1. computes dot product of previous input with individual connection weights to obtain each neuron output
		2. add neuron bias to each neuron output
		3. applies activation function to each neuron (if set)
		"""
		weights = np.array([neuron.weights for neuron in self.neurons]) # shape = (neuron_count, input_feature_size)
		mul_outcome = np.matmul(input_feature, weights.T) # (input_feature_size,) * (input_feature_size, neuron_count) = (neuron_count,)

		biases = [neuron.bias for neuron in self.neurons] # 1D np array, same shape as mul_outcome (i.e. shape = (neuron_count,))
		add_outcome = mul_outcome +np.array(biases) # mul_outcome and biases have the same shape

		if (hasattr(self, "activation")):
			return self.activation(add_outcome)
		return add_outcome

class DenseLayer(Layer):
	pass

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

	def outcome(self, input_feature):
		"""
		no weights and biases to apply function to
		"""
		return input_feature

class NeuralNetwork():
	"""
	contains layers of neuron and activation function
	"""
	def __init__(self, layers, learning_rate=0.01):
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
		for epoch in range(250):
			total_loss = 0 # use mean squared residuals (mse) as loss function
			predictions = []

			for sample_idx, sample in enumerate(x):
				# feedforward
				activations = [] # store activations to help compute loss function derivatives with respect to the individual weights
				current_sample_activation = []
				for layer in self.layers:
					if (len(activations) >= 1):
						# subsequent layer, pass in output from previous layer as the input to this current layer
						activations.append(layer.outcome(activations[-1]))
					else:
						# first layer, pass in input
						activations.append(layer.outcome(sample))

				# compute total loss
				# activations[-1] stores the output of the final layer, aka prediction output
				loss = ((y[sample_idx] -activations[-1]) **2).mean() # mse
				total_loss += loss

				# dE/dY, derivative of loss function (error) with respect to prediction output
				# sum of squared residuals prime = 2(y-y^), where y is predicted and y^ is ground truth
				loss_derivative_output = np.array([2 *(activations[-1] -y[sample_idx]) /y[sample_idx].size]) # wrap in brackets to convert from 1D array to 2D array (important for computing dot product later)

				# backpropagation
				# start from outer most layer - output layer first (because dE/dX for the subsequent layer can be computed by the current dE/dY for the current layer, where X is the input and Y is the output)
				for i in range(len(self.layers) -1, 0, -1): # dont traverse input layer (first layer) since there are no weights there
					layer = self.layers[i]

					# compute dE/dW, derivative of loss function (error) with respect to individual weights in layer
					# where W.shape = (output_neuron, input_neuron)

					# goal is to find the rate of change (differentiaton) of the loss function (error) with respect to the weight
					# work out entire neural network equation to differentiate is not feasible
					# hence using chain rule, dE/dw(ij) = dE/dy1 * dy1/dw(ij) + ... + dE/dy1 * dyj/dw(ij), where i is number of input neruons, j is number of output neurons
					# dE/dW = dot_product(X^t, dE/dY)
					# localised to only one layer at a time
					X = np.array([activations[i -1]]) # wrap in brackets to convert from 1D array to 2D array
					loss_derivative_weights = X.T @ loss_derivative_output # dot product of both matrix

					# do the same for bias
					# dE/dB = [dE/db1, dE/db2, ..., dE/db(j)], where j is the number of output neurons
					# dE/db(j) = dE/dy1 * dy1/db(j) + ... + dE/dy(j) * dy(j)/db(j) = dE/dy(j) = dE/dY
					loss_derivative_biases = loss_derivative_output

					# compute dE/dX to be used by the previous layer
					# dE/dX = [dE/dx1, ..., dE/dx(i)], where i is the number of input neurons
					loss_derivative_input = loss_derivative_output @ loss_derivative_weights.T

					# adjust weights
					for neuron_idx, weight_error in enumerate(loss_derivative_weights.T):
						layer.neurons[neuron_idx].weights -= weight_error *self.learning_rate
						layer.neurons[neuron_idx].bias -= loss_derivative_biases.T[neuron_idx][0] *self.learning_rate

					# set output
					loss_derivative_output = loss_derivative_input # where dE/dX is the output of the next previous layer we will traverse to

				predictions.append(activations[-1])

			# compute average loss
			print("avg loss", total_loss /y.size)

		# todo: implement backpropagation
		return predictions




