import numpy as np

class LogisticRegression():
	def __init__(self, iterations_limit=1000, learning_rate=0.05, threshold=0.5):
		self.iterations_limit = iterations_limit
		self.learning_rate = learning_rate
		self.threshold = threshold # probability threshold (inclusive) for positive class (e.g. >= 0.5 classified as positive)

	def fit(self, X, y):
		"""
		X: numpy.ndarray, 2d array of input feautres
		y: numpy.ndarray, 1d array consisting of target labels (1s and 0s only)
		"""
		self.weights = np.ones(X.shape[1]) # .shape() => [number of observations, number of features]
		self.intercept = 0

		for iter_idx in range(self.iterations_limit):
			summed_neg_log_likelihood = 0
			for idx, input_feature in enumerate(X):
				z = self.weights.dot(input_feature).sum() +self.intercept # compute logit w*x + b
				# y_pred = sigmoid(z)
				y_pred = 1 /(1 +np.exp(-z)) # pass it into sigmoid function to return y_pred (y^)

				# y_truth is the truth label
				y_truth = y[idx]

				# compute derivative of loss function (negative log likelihood) with respect to each weights
				# p(y_pred=y_truth|x; z) = y_pred^(y_truth) * (1 - y_pred)^(1 - y_truth), where y_truth is in a bernoulli distro, i.e. 0 and 1 only)
				# likelihood = p(y_pred=y_truth|x; z); product of individual probabilites when computed across all independent observations, hence when log, we can take summation instead; also taking log will increase penalty for more confident mistakes
				# log likelihood = y_truth * ln(y_pred) + (1 - y_truth) * ln(1 - y_pred), where ln(ab) = ln(a) + ln(b)
				# deriviative of likelihood with respect to w_j = [y_pred-y_truth]*x_j (where x_j is the jth input feature)
				# minimise negative log likelihood (convex shape, derivative should approach 0 - minimise)
				derivative_weights = (y_pred -y_truth) *input_feature # derivative of negative log likelihood; same shape as self.weights
				derivative_intercept = (y_pred -y_truth) # simply swap out dz/dw_j with dz/dc, where c is the intercept

				# adjust weights and intercept according to gradient computed
				self.weights -= derivative_weights *self.learning_rate
				self.intercept -= derivative_intercept *self.learning_rate

				# add neg log likelihood to sum
				summed_neg_log_likelihood += -(y_truth *np.log(y_pred) +(1 -y_truth) *np.log(1 -y_pred))

			# determine stopping threshold
			if (summed_neg_log_likelihood /X.shape[0] <= 0.05): # averaged by number of observations
				# stop training
				return

	def predict(self, X):
		"""
		X: numpy.ndarray, 2d array of input feautres
		"""
		output_labels = []
		for input_feature in X:
			z = self.weights.dot(input_feature).sum() +self.intercept
			sig = 1 /(1 + np.exp(-z))
			output_labels.append(1 if sig >= self.threshold else 0)
		return output_labels

if __name__ == "__main__":
	# demo code
	# learn data based on bitwise AND, linearly separable
	lr = LogisticRegression()
	X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1], [1, 1]])
	y = np.array([0, 0, 0, 1, 1, 1])
	lr.fit(X, y)
	print(lr.predict([[1, 1]]))
	print(lr.weights)
