import random
import torch, torch.nn.Functional as F

class SingleHeadAttention:
	def __init__(self):
		pass

	def fit(self, X):
		"""
		X: torch.tensor, matrix where each row contains the embedding for i-th document, shape <T, d>, where T is the number of documents, d is the dimensionality of the input embeddings
		"""
		# seed torch's RNG
		self.seed = random.randint(1, 10)
		torch.manual_seed(self.seed)

		self.X = X

		self.T = X.shape[0] # number of documents
		self.d = X.shape[1] # input embedding dimensionality
		self.d_k = 5 # arbitrary number, common dimension between query and key vectors
		self.d_v = 6 # arbitrary number, output embedding dimensionality

		self.Wq = torch.rand(self.d_k, d) # shape <d_k, d>, where d_k is the same for self.Wk shape, d_k is an arbitrary number chosen to represent number of dimensionality between Q and K when computing dot product
		self.Wk = torch.rand(self.d_k, d)

		self.Wv = torch.rand(self.d_v, d)

	def forward(self):
		"""
		forward pass

		let A be the self-attention matrix, shape <T, T>
		Q = X * Wq^T, where each row in matrix Q represents the query vector for each document
		K = X * Wk^T, where each row in matrix K represents the key vector for each document
		Z = (Q * K^T)/(sqrt(d_k)), where numerator is the dot product between matrix Q and matrix K transposed
		A = row-wise softmax(Z)

		V = X * Wv^T, where each row in matrix V represents the value vector for each document
		final embedding = A * V
		"""
		Q = self.X.matmul(self.Wq.transpose())
		K = self.X.matmul(self.Wk.transpose())

		A = Q.amtmul(K.transpose())

		# scale by 1/sqrt(d_k), standardise weights so can exist on same magnitude, easier convergence
		A_scaled = A /(self.d_k) **.5

		# compute row-wise softmax to obtain normalised attention weights
		normalised_weights = F.softmax(A_scaled, dim=0)

		# compute context vector (attention weights * V)
		return normalised_weights.matmul(A)

	def backward(self):
		"""
		backward pass, propagation
		"""
		pass