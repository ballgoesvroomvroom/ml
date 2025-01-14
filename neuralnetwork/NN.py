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
		weights = np.random.rand(input_size)
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
	def __init__(self, layers, learning_rate=0.1):
		"""
		layers: Layer[], contains the layers for the network, first and last element contains the input and output layer; therefore, min length for layers is 2 
		learning_rate: float, influences step size when learning weights
		"""
		self.layers = layers
		self.learning_rate = 0

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

		# feedforward
		pred = []
		for sample in x:
			prev = sample
			for layer in self.layers:
				prev = layer.outcome(prev)
				print("prev", prev)
			pred.append(prev)

		# todo: implement backpropagation
		return pred




