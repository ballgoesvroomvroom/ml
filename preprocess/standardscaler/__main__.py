"""
scale input features to have mean of 0 and standard deviation of 1
"""
class StandardScaler():
	def __init__(self):
		pass

	def fit(self, X, y=None):
		"""
		X: numpy.ndarray, 2D array containing input features

		subtract feature's mean respectively and divide by feature's standard deviation respectively

		returns self for chaining (e.g. .fit().transform())
		"""
		self.features_mean = X.mean(axis=0) # mean for each column (feature)
		self.features_std = X.std(axis=0) # standard deviation for each column (feature)

		return self


	def transform(self, X):
		"""
		X: numpy.ndarray, 2D array containing input features

		subtract learnt feature's mean respectively and divide by learnt feature's standard deviation respectively

		returns scaled_features: numpy.ndarray, with same shape as X
		"""
		return (X -self.features_mean) /self.features_std