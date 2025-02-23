"""
Byte-pair encoding (BPE) to tokenize
"""
import re
import string

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("punkt_tab")
nltk.download("wordnet")


def pre_tokenize(doc):
	# doc: string

	# lowercase
	doc = doc.lower()

	# split into tokens
	tokens = [
	    token for sent in sent_tokenize(doc) for token in word_tokenize(sent)
	]

	# remove punctuations
	non_alphabetical = re.compile("[^a-zA-Z]")
	stripped_punct_tokens = [
	    non_alphabetical.sub("", token) for token in tokens
	]

	# remove tokens with len() <= 2 (character count)
	min_twochar_tokens = [
	    token for token in stripped_punct_tokens if len(token) >= 3
	]

	# lemmatise tokens
	lemmatiser = WordNetLemmatizer()
	lemmatised = [lemmatiser.lemmatize(token) for token in min_twochar_tokens]

	return lemmatised


class BytePairEncoder:

	def __init__(self):
		pass

	def common_pairs(self):
		counter = {}
		for idx, char in enumerate(self.chars):
			if (idx >= len(self.chars) - 1):
				# last one ignore
				continue

			pair = (char, self.chars[idx + 1])
			if (pair in counter):
				counter[pair] += 1
			else:
				counter[pair] = 1
		return counter

	def fit(self, tokens):
		# tokens: str[], pre-processed tokens
		# obtain characters
		self.chars = [char for token in tokens for char in str(token)]

		# build vocab characters
		self.vocab = set(self.chars)

		# find merge rules
		self.merge_rules = []
		for i in range(200):
			# 10 iterations
			pairs = self.common_pairs()

			# get max token
			most_common_pair = None  # key element of return value (dict) self.common_pairs()
			for pair, freq in pairs.items():
				if (most_common_pair and freq >= pairs[most_common_pair]):
					most_common_pair = pair
				elif (most_common_pair is None):
					# default first item as baseline
					most_common_pair = pair

			# append to merge rules
			self.merge_rules.append(most_common_pair)

			# merge chars
			new_chars = []
			local_context = None
			for idx, char in enumerate(self.chars):
				if (char == most_common_pair[0]):
					# start of a pair
					if (local_context):
						new_chars.append(local_context)
					local_context = char
				elif (char == most_common_pair[1]
				      and local_context is not None):
					# end of a pair
					new_chars.append(local_context + char)
					local_context = None  # reset state
				else:
					# not a pair
					if (local_context):
						new_chars.append(local_context)
						local_context = None  # reset state
					new_chars.append(char)
			self.chars = new_chars  # set chars

		return self.chars


corpus = pd.read_csv(
    "preprocess/text/bytepair_encoding/training_corpus/sentences.txt",
    sep="\t",
    names=["index", "text"])
print(corpus)
tokens = list(corpus["text"].apply(pre_tokenize).explode())
bpe = BytePairEncoder()
print(bpe.fit(tokens))
# tokens = pre_tokenize("Hello world. Mrs. Smith wants some granny apples. Please get her some apples")
# # tokens = pre_tokenize("Mrs. Smith wants some granny apples.")
# print(tokens)
# bpe = BytePairEncoder()
# print(bpe.fit(tokens))
