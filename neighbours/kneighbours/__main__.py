class KNeighboursClassifier():
	def __init__(self, k=5, distance="euclidean", p=None):
		self.distance = distance # "euclidean"|"manhattan"|"minkowski"
		self.k = k
		self.p = p # only applicable if distance == "minkowski", otherwise ignored

	def set_params(self, params):
		"""
		params: {
			k: number,
			distance: "euclidean"|"manhattan"|"minkowski",
			p: number, only applicable if distance == "minkowski", otherwise ignored
		}
		"""
		if ("k" in params):
			self.k = params["k"]
		if ("distance" in params):
			self.distance = params["distance"]
		if ("p" in params):
			self.p = params["p"]

	def get_params(self):
		return {
			"k": self.k,
			"distance": self.distance,
			"p": self.p
		}

	def fit(self, X, y):
		"""
		X: numpy.ndarray, input observations
		y: numpy.ndarray, outcome labels in 1s and 0s

		returns self for chaining (e.g. .fit().transform())
		"""
		self.observations = X
		self.truth_labels = y

		self._classes, _ = np.unique(y, return_counts=True) # values, counts

		return self # for chaining, call .transform() immediately

	def transform(self, X, y=None):
		"""
		X: numpy.ndarray, input observations

		returns predicted_labels: numpy.ndarray, outcome labels in 1s and 0s
		"""
		predicted_labels = []
		distance_fn = self._dist_euclidean if self.distance == "euclidean" else \
						(self._dist_manhattan if self.distance == "manhattan" else \
							lambda a, b: self._dist_minkowski(a, b, self.p))
		for input_observation in X:
			distances = [distance_fn(input_observation, training_observation) for training_observation in self.observations]

			# obtain indices of top k neighbours (minimum distances)
			top_k_neighbours = np.argsort(distances)[:self.k]

			# get class frequency of truth labels of selected neighbours
			top_k_neighbours_labels, counts = np.unique(self.truth_labels[top_k_neighbours], return_counts=True)
			highest_truth_label_idx = np.argmax(counts)

			# index truth label (unique list) to obtain majority class amongst top k neighbours
			y_pred = top_k_neighbours_labels[highest_truth_label_idx]
			predicted_labels.append(y_pred)
		return np.array(predicted_labels)

	def _dist_euclidean(self, a, b):
		"""
		a: numpy.ndarray, vector representing coordinate point of first observation
		b: numpy.ndarray, vector representing coordinate point of second observation

		returns the euclidean distance between two points a and b
		"""
		return np.sqrt(np.power((b -a), 2).sum())

	def _dist_manhattan(self, a, b):
		"""
		a: numpy.ndarray, vector representing coordinate point of first observation
		b: numpy.ndarray, vector representing coordinate point of second observation

		returns the manhanttan distance between two points a and b
		"""
		return np.abs(a -b).sum()

	def _dist_minkowski(self, a, b, p=3):
		"""
		a: numpy.ndarray, vector representing coordinate point of first observation
		b: numpy.ndarray, vector representing coordinate point of second observation
		p: integer, 

		returns the minkowski distance between two points a and b
		"""
		return np.power(np.power(np.abs(a -b), p).sum(), 1 /p)